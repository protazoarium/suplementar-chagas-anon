"""Testes da avaliação dos modelos (AUC-ROC e curvas ROC)."""

from pathlib import Path

import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier

from tcc_datasus.evaluate import (
    CV_REPORT_FILENAME,
    FEATURE_IMPORTANCE_PLOT_FILENAME,
    ROC_PLOT_FILENAME,
    SUMMARY_REPORT_FILENAME,
    TUNING_COMPARISON_FILENAME,
    compare_models,
    compare_tuning,
    cross_validate_auc_roc,
    evaluate_auc_roc,
    plot_feature_importance,
    plot_roc_curves,
    report_cross_validation,
    write_summary_report,
)
from tcc_datasus.train import train_random_forest

# Dataset sintético pequeno, mas separável o suficiente para um AUC-ROC
# alto e determinístico (feature única correlacionada com o rótulo).
X_TREINO = pl.DataFrame({"x": [0, 1, 2, 3, 10, 11, 12, 13]})
Y_TREINO = pl.Series("label", [0, 0, 0, 0, 1, 1, 1, 1])
X_TESTE = pl.DataFrame({"x": [0.5, 2.5, 10.5, 12.5]})
Y_TESTE = pl.Series("label", [0, 0, 1, 1])

# Datamart sintético para cross_validate_auc_roc: 3 UFs x 100
# município-anos, 10% de risco por UF -- acima do mínimo exigido por
# StratifiedKFold com n_splits=5 (ver tests/test_train_split.py).
N_POR_UF_CV = 100
DATAMART_SINTETICO_CV = pl.DataFrame(
    {
        "uf": [uf for uf in ("PE", "PB", "AL") for _ in range(N_POR_UF_CV)],
        "ano": list(range(2000, 2000 + N_POR_UF_CV)) * 3,
        "casos_lag_1": [0] * (N_POR_UF_CV * 3),
        "casos_lag_2": [0] * (N_POR_UF_CV * 3),
        "populacao": [1_000_000] * (N_POR_UF_CV * 3),
        "risco": [1 if i % 10 == 0 else 0 for i in range(N_POR_UF_CV)] * 3,
    }
)


@pytest.fixture
def modelo_treinado() -> RandomForestClassifier:
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_TREINO.to_numpy(), Y_TREINO.to_numpy())
    return modelo


def test_evaluate_auc_roc_dataset_separavel_da_auc_alta(modelo_treinado: RandomForestClassifier):
    auc = evaluate_auc_roc(modelo_treinado, X_TESTE, Y_TESTE)
    assert auc == pytest.approx(1.0)


def test_compare_models_aponta_o_maior_auc_como_melhor(tmp_path: Path):
    resultados = {"Random Forest": 0.85, "XGBoost": 0.90}
    caminho = compare_models(resultados, tmp_path)
    conteudo = caminho.read_text(encoding="utf-8")
    assert "Melhor modelo: **XGBoost**" in conteudo
    assert "0.9000" in conteudo
    assert "0.8500" in conteudo


def test_plot_roc_curves_gera_png_valido(tmp_path: Path, modelo_treinado: RandomForestClassifier):
    caminho = plot_roc_curves({"Random Forest": modelo_treinado}, X_TESTE, Y_TESTE, tmp_path)
    assert caminho == tmp_path / ROC_PLOT_FILENAME
    assert caminho.exists()
    assert caminho.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_feature_importance_gera_png_valido(tmp_path: Path):
    importancias = {
        "Random Forest": pl.DataFrame({"feature": ["x", "y"], "importancia": [0.7, 0.3]}),
        "XGBoost": pl.DataFrame({"feature": ["x", "y"], "importancia": [0.6, 0.4]}),
    }
    caminho = plot_feature_importance(importancias, tmp_path)
    assert caminho == tmp_path / FEATURE_IMPORTANCE_PLOT_FILENAME
    assert caminho.exists()
    assert caminho.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cross_validate_auc_roc_retorna_uma_auc_por_fold():
    scores = cross_validate_auc_roc(train_random_forest, DATAMART_SINTETICO_CV, n_splits=5)
    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_report_cross_validation_aponta_a_maior_media_como_melhor(tmp_path: Path):
    resultados = {"Random Forest": [0.80, 0.85, 0.90], "XGBoost": [0.70, 0.75, 0.72]}
    caminho = report_cross_validation(resultados, tmp_path)
    assert caminho == tmp_path / CV_REPORT_FILENAME
    conteudo = caminho.read_text(encoding="utf-8")
    assert "Número de folds: 3" in conteudo
    assert "Melhor modelo: **Random Forest**" in conteudo
    assert "0.8500" in conteudo  # média de Random Forest


def test_report_cross_validation_um_fold_nao_quebra_no_desvio_padrao(tmp_path: Path):
    resultados = {"Random Forest": [0.9]}
    caminho = report_cross_validation(resultados, tmp_path)
    conteudo = caminho.read_text(encoding="utf-8")
    assert "0.9000 ± 0.0000" in conteudo


def test_compare_tuning_mostra_delta_positivo_quando_ajuste_melhora(tmp_path: Path):
    padrao = {"Random Forest": [0.80, 0.82, 0.84], "XGBoost": [0.70, 0.72, 0.74]}
    ajustado = {"Random Forest": [0.85, 0.87, 0.89], "XGBoost": [0.68, 0.70, 0.72]}
    caminho = compare_tuning(padrao, ajustado, tmp_path)

    assert caminho == tmp_path / TUNING_COMPARISON_FILENAME
    conteudo = caminho.read_text(encoding="utf-8")
    assert "0.8200 ± 0.0200" in conteudo  # média/desvio padrão de Random Forest
    assert "0.8700 ± 0.0200" in conteudo  # média/desvio ajustado de Random Forest
    assert "+0.0500" in conteudo  # delta de Random Forest (melhora)
    assert "-0.0200" in conteudo  # delta de XGBoost (piora)


def test_compare_tuning_ignora_modelos_sem_par_nos_dois_conjuntos(tmp_path: Path):
    padrao = {"Random Forest": [0.80, 0.82]}
    ajustado = {"Random Forest": [0.85, 0.87], "XGBoost": [0.90, 0.92]}
    caminho = compare_tuning(padrao, ajustado, tmp_path)
    conteudo = caminho.read_text(encoding="utf-8")
    assert "Random Forest" in conteudo
    assert "XGBoost" not in conteudo


def test_write_summary_report_consolida_as_quatro_secoes(tmp_path: Path):
    single_split = {"Random Forest": 0.93, "XGBoost": 0.89}
    tuned_cv = {"Random Forest": [0.90, 0.92, 0.94], "XGBoost": [0.88, 0.90, 0.89]}
    baseline_cv = {"Random Forest": [0.85, 0.87, 0.86], "XGBoost": [0.80, 0.82, 0.81]}
    importancias = {
        "Random Forest": pl.DataFrame({"feature": ["ano", "uf", "casos_lag_1"], "importancia": [0.5, 0.3, 0.2]}),
        "XGBoost": pl.DataFrame({"feature": ["ano", "uf", "casos_lag_1"], "importancia": [0.4, 0.35, 0.25]}),
    }

    caminho = write_summary_report(single_split, tuned_cv, baseline_cv, importancias, tmp_path)
    assert caminho == tmp_path / SUMMARY_REPORT_FILENAME
    conteudo = caminho.read_text(encoding="utf-8")

    assert "## Desempenho final (validação cruzada, hiperparâmetros ajustados)" in conteudo
    assert "## Efeito da busca de hiperparâmetros" in conteudo
    assert "## AUC-ROC de uma única divisão treino/teste (referência)" in conteudo
    assert "## Features mais importantes" in conteudo
    assert "Melhor modelo: Random Forest" in conteudo
    assert "ano (0.500)" in conteudo


def test_write_summary_report_referencia_os_outros_arquivos_de_relatorio(tmp_path: Path):
    single_split = {"Random Forest": 0.93}
    tuned_cv = {"Random Forest": [0.90, 0.92]}
    baseline_cv = {"Random Forest": [0.85, 0.87]}
    importancias = {"Random Forest": pl.DataFrame({"feature": ["ano"], "importancia": [1.0]})}

    caminho = write_summary_report(single_split, tuned_cv, baseline_cv, importancias, tmp_path)
    conteudo = caminho.read_text(encoding="utf-8")

    for nome_arquivo in (
        CV_REPORT_FILENAME,
        TUNING_COMPARISON_FILENAME,
        FEATURE_IMPORTANCE_PLOT_FILENAME,
        ROC_PLOT_FILENAME,
    ):
        assert nome_arquivo in conteudo
