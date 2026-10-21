"""Testes da inferência (carregar um modelo treinado e pontuar município-anos)."""

from pathlib import Path

import polars as pl
import pytest

from tcc_datasus.predict import COLUNA_PREVISAO_PADRAO, COLUNA_PROBABILIDADE_PADRAO, load_model, predict_risk
from tcc_datasus.train import encode_features, save_model, train_random_forest

# Datamart sintético pequeno, mas separável o suficiente para
# previsões determinísticas (município-anos de PE sempre de risco,
# de AL nunca).
DATAMART_TREINO = pl.DataFrame(
    {
        "uf": ["PE"] * 5 + ["AL"] * 5,
        "ano": list(range(2015, 2020)) * 2,
        "casos_lag_1": [3] * 5 + [0] * 5,
        "casos_lag_2": [3] * 5 + [0] * 5,
        "populacao": [1_000_000] * 10,
        "risco": [1] * 5 + [0] * 5,
    }
)


@pytest.fixture
def modelo_treinado():
    X = encode_features(DATAMART_TREINO)
    y = DATAMART_TREINO["risco"]
    return train_random_forest(X, y)


def test_predict_risk_adiciona_colunas_e_preserva_originais(modelo_treinado):
    previsoes = predict_risk(modelo_treinado, DATAMART_TREINO)

    assert COLUNA_PROBABILIDADE_PADRAO in previsoes.columns
    assert COLUNA_PREVISAO_PADRAO in previsoes.columns
    for coluna in DATAMART_TREINO.columns:
        assert coluna in previsoes.columns
    assert len(previsoes) == len(DATAMART_TREINO)


def test_predict_risk_probabilidades_no_intervalo_valido(modelo_treinado):
    previsoes = predict_risk(modelo_treinado, DATAMART_TREINO)
    assert previsoes[COLUNA_PROBABILIDADE_PADRAO].is_between(0.0, 1.0).all()
    assert set(previsoes[COLUNA_PREVISAO_PADRAO].unique().to_list()) <= {0, 1}


def test_predict_risk_nao_exige_coluna_de_rotulo(modelo_treinado):
    sem_risco = DATAMART_TREINO.drop("risco")
    previsoes = predict_risk(modelo_treinado, sem_risco)
    assert COLUNA_PROBABILIDADE_PADRAO in previsoes.columns


def test_predict_risk_respeita_nomes_de_coluna_customizados(modelo_treinado):
    previsoes = predict_risk(
        modelo_treinado,
        DATAMART_TREINO,
        probabilidade_col="prob_rf",
        previsao_col="previsto_rf",
    )
    assert "prob_rf" in previsoes.columns
    assert "previsto_rf" in previsoes.columns


def test_load_model_roundtrip_com_save_model(tmp_path: Path, modelo_treinado):
    caminho = tmp_path / "modelo.joblib"
    save_model(modelo_treinado, caminho)

    modelo_carregado = load_model(caminho)
    previsoes_original = predict_risk(modelo_treinado, DATAMART_TREINO)
    previsoes_carregado = predict_risk(modelo_carregado, DATAMART_TREINO)

    assert (
        previsoes_original[COLUNA_PROBABILIDADE_PADRAO].to_list()
        == previsoes_carregado[COLUNA_PROBABILIDADE_PADRAO].to_list()
    )
