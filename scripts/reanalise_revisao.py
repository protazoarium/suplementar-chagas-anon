"""Reanálise técnica solicitada pelos avaliadores do SBTI 2026 (ver sugestoes.md).

Script único e não reutilizado pelo pipeline principal (run_etl.py/
run_train.py) -- roda as análises adicionais pedidas na revisão por cima
do datamart já corrigido (grade município-ano oficial + população
municipal, ver `transform.aggregate_municipio_ano_lags`/
`fetch_ibge_population_municipal`): validação temporal com/sem a feature
"ano", baselines de referência, validação cruzada aninhada, métricas
complementares (AUC-PR, top-k, calibração) e importância por permutação.

Não altera `data/datamart/` nem os modelos salvos em `models/` (que
continuam sendo os de `run_train.py`) -- só lê o datamart e escreve
relatórios novos/atualizados em `reports/`.
"""

from functools import partial
from pathlib import Path

import polars as pl

from tcc_datasus.evaluate import (
    calibration_metrics,
    cross_validate_auc_roc,
    evaluate_auc_pr,
    evaluate_auc_roc,
    permutation_feature_importance,
    plot_permutation_importance,
    report_baselines,
    report_nested_cv,
    report_temporal_validation,
    top_k_precision_recall,
)
from tcc_datasus.train import (
    RegraCasosAnoAnterior,
    load_datamart,
    split_train_test,
    temporal_splits,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    tune_random_forest,
    tune_xgboost,
)

DATAMART_DIR = Path("data/datamart")
REPORTS_DIR = Path("reports")


def main() -> None:
    datamart = load_datamart(DATAMART_DIR)
    print(f"Datamart: {datamart.height} município-anos, {datamart['risco'].sum()} de risco")

    X_train, X_test, y_train, y_test = split_train_test(datamart)

    print("Ajustando RF/XGBoost (hiperparâmetros, divisão única)...")
    rf = tune_random_forest(X_train, y_train)
    xgb = tune_xgboost(X_train, y_train)
    rf_params = {k: v for k, v in rf.get_params().items()}
    xgb_params = {k: v for k, v in xgb.get_params().items()}

    # --- Baselines (divisão única, mesmo X_test/y_test de RF/XGBoost) ---
    print("Avaliando baselines (divisão única)...")
    regra = RegraCasosAnoAnterior().fit(X_train.to_numpy(), y_train.to_numpy())
    logreg_completa = train_logistic_regression(X_train, y_train)
    logreg_ano_uf = train_logistic_regression(X_train, y_train, colunas=["ano", "UF_COD"])

    modelos_e_baselines = {
        "Random Forest": rf,
        "XGBoost": xgb,
        "Baseline: casos do ano anterior": regra,
        "Baseline: regressão logística (todas as features)": logreg_completa,
        "Baseline: regressão logística (ano + UF)": logreg_ano_uf,
    }
    resultados_baseline = {
        nome: {"auc_roc": evaluate_auc_roc(m, X_test, y_test), "auc_pr": evaluate_auc_pr(m, X_test, y_test)}
        for nome, m in modelos_e_baselines.items()
    }
    for nome, m in resultados_baseline.items():
        print(f"  {nome}: AUC-ROC={m['auc_roc']:.4f} AUC-PR={m['auc_pr']:.4f}")
    report_baselines(resultados_baseline, REPORTS_DIR)

    # --- Métricas complementares (RF/XGBoost, divisão única) ---
    print("Métricas complementares (top-k, calibração)...")
    linhas_complementares = ["# Métricas complementares (AUC-PR, top-k, calibração)", ""]
    for nome, m in (("Random Forest", rf), ("XGBoost", xgb)):
        topo = top_k_precision_recall(m, X_test, y_test)
        brier, curva = calibration_metrics(m, X_test, y_test)
        linhas_complementares += [f"## {nome}", "", f"Brier score: {brier:.4f}", "", "Top-k:", ""]
        linhas_complementares.append(topo.write_csv())
        linhas_complementares += ["Curva de calibração:", "", curva.write_csv()]
    (REPORTS_DIR / "metricas_complementares.md").write_text("\n".join(linhas_complementares), encoding="utf-8")

    # --- Importância por permutação (RF/XGBoost, conjunto de teste) ---
    print("Importância por permutação...")
    importancias_perm = {
        "Random Forest": permutation_feature_importance(rf, X_test, y_test),
        "XGBoost": permutation_feature_importance(xgb, X_test, y_test),
    }
    for nome, df in importancias_perm.items():
        print(f"  {nome}:\n{df}")
    plot_permutation_importance(importancias_perm, REPORTS_DIR)
    for nome, df in importancias_perm.items():
        df.write_csv(REPORTS_DIR / f"importancia_permutacao_{nome.lower().replace(' ', '_')}.csv")

    # --- CV ajustada (não aninhada, como no pipeline original) vs. aninhada ---
    print("Validação cruzada: não aninhada vs. aninhada (pode levar alguns minutos)...")
    cv_nao_aninhada = {
        "Random Forest": cross_validate_auc_roc(partial(train_random_forest, **rf_params), datamart),
        "XGBoost": cross_validate_auc_roc(partial(train_xgboost, **xgb_params), datamart),
    }
    cv_aninhada = {
        "Random Forest": cross_validate_auc_roc(tune_random_forest, datamart),
        "XGBoost": cross_validate_auc_roc(tune_xgboost, datamart),
    }
    print("  Não aninhada:", cv_nao_aninhada)
    print("  Aninhada:", cv_aninhada)
    report_nested_cv(cv_nao_aninhada, cv_aninhada, REPORTS_DIR)

    # --- Validação temporal (origem deslizante), com e sem "ano" ---
    print("Validação temporal (origem deslizante)...")
    resultados_temporais: dict[str, list[tuple[int, float, float]]] = {}
    for remover_ano in (False, True):
        sufixo = " (sem 'ano')" if remover_ano else " (com 'ano')"
        for nome_modelo, treinar in (
            ("Random Forest" + sufixo, partial(train_random_forest, **rf_params)),
            ("XGBoost" + sufixo, partial(train_xgboost, **xgb_params)),
        ):
            splits_resultado = []
            for X_tr, X_te, y_tr, y_te, ano_teste in temporal_splits(datamart, remover_ano=remover_ano):
                modelo = treinar(X_tr, y_tr)
                auc_roc = evaluate_auc_roc(modelo, X_te, y_te)
                auc_pr = evaluate_auc_pr(modelo, X_te, y_te)
                splits_resultado.append((ano_teste, auc_roc, auc_pr))
            resultados_temporais[nome_modelo] = splits_resultado
            media = sum(a for _, a, _ in splits_resultado) / len(splits_resultado)
            print(f"  {nome_modelo}: AUC-ROC média = {media:.4f} ({len(splits_resultado)} splits)")
    report_temporal_validation(resultados_temporais, REPORTS_DIR)

    print("Concluído. Relatórios em reports/.")


if __name__ == "__main__":
    main()
