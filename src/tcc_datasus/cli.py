"""CLI e funções ponta a ponta do pipeline (etl, treino, inferência).

Cada função aqui é a implementação real por trás tanto de `uv run
tcc-datasus <comando>` quanto dos scripts na raiz do projeto
(`run_etl.py`, `run_train.py`, `run_predict.py`, que só chamam essas
funções) -- um único lugar de manutenção para as duas formas de rodar o
pipeline.
"""

import argparse
from functools import partial
from pathlib import Path

from tcc_datasus.convert import dbc_to_dbf, dbf_to_dataframe
from tcc_datasus.download import download_all, list_available_years
from tcc_datasus.evaluate import (
    compare_models,
    compare_tuning,
    cross_validate_auc_roc,
    evaluate_auc_roc,
    plot_feature_importance,
    plot_roc_curves,
    report_cross_validation,
    write_summary_report,
)
from tcc_datasus.features import apply_epidemiological_filters, impute_missing_values
from tcc_datasus.predict import load_model, predict_risk
from tcc_datasus.train import (
    get_feature_importance,
    load_datamart,
    save_model,
    split_train_test,
    train_random_forest,
    train_xgboost,
    tune_random_forest,
    tune_xgboost,
)
from tcc_datasus.transform import (
    build_risk_datamart,
    fetch_ibge_population_municipal,
    prepare_partition_columns,
    write_datamart_parquet,
    write_partitioned_parquet,
)

UFS = ["PE", "PB", "AL"]
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
DATAMART_DIR = Path("data/datamart")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

RANDOM_FOREST_FILENAME = "random_forest.joblib"
XGBOOST_FILENAME = "xgboost.joblib"
PREDICTIONS_FILENAME = "previsoes_risco.csv"


def run_etl() -> None:
    """Baixa, converte e transforma o SINAN-DCA (PE/PB/AL) em data/processed/ e data/datamart/.

    Cobre todo o intervalo de fato publicado pelo DATASUS: `FINAIS`
    (dados consolidados) e, para os anos que só existem lá, `PRELIM`
    (sujeitos a revisão) -- ver `download.FTP_DIRS`. Um ano que aparece
    nas duas fontes (ex.: se um ano de `PRELIM` for promovido a `FINAIS`
    numa publicação futura do DATASUS) usa sempre a versão `FINAIS`; se
    o `.dbc` de `PRELIM` já tiver sido baixado localmente antes dessa
    promoção, `download_dbc`/`download_all` não o baixam de novo por
    padrão (`overwrite=False`) -- rode com `overwrite=True` manualmente
    nesse caso raro.
    """
    anos_finais = list_available_years("FINAIS")
    anos_prelim = [ano for ano in list_available_years("PRELIM") if ano not in anos_finais]

    for fonte, anos in (("FINAIS", anos_finais), ("PRELIM", anos_prelim)):
        for p in download_all(anos, RAW_DIR, source=fonte):
            dbf = dbc_to_dbf(p, INTERIM_DIR)
            df = dbf_to_dataframe(dbf, ufs=UFS)
            lf = impute_missing_values(apply_epidemiological_filters(df.lazy()))
            write_partitioned_parquet(prepare_partition_columns(lf.collect()), PROCESSED_DIR)

    todos_anos = anos_finais + anos_prelim
    populacao = fetch_ibge_population_municipal(ano_inicio=min(todos_anos), ano_fim=max(todos_anos))
    write_datamart_parquet(build_risk_datamart(PROCESSED_DIR, populacao=populacao), DATAMART_DIR)


def run_train() -> None:
    """Divide, ajusta hiperparâmetros, treina, avalia e salva os modelos.

    Requer `data/datamart/` já populado (ver `run_etl`). Salva os dois
    modelos ajustados em `models/` e os relatórios de avaliação em
    `reports/` (comparação de uma única divisão, validação cruzada,
    comparação padrão vs. ajustado, importância das features, e o
    resumo consolidado).
    """
    datamart = load_datamart(DATAMART_DIR)
    X_train, X_test, y_train, y_test = split_train_test(datamart)

    # tune_* já treina com os melhores hiperparâmetros encontrados (grid
    # search com cv estratificado por UF+rótulo, ver train.py) --
    # substitui diretamente train_random_forest/train_xgboost.
    random_forest = tune_random_forest(X_train, y_train)
    xgboost_model = tune_xgboost(X_train, y_train)

    save_model(random_forest, MODELS_DIR / RANDOM_FOREST_FILENAME)
    save_model(xgboost_model, MODELS_DIR / XGBOOST_FILENAME)

    modelos = {"Random Forest": random_forest, "XGBoost": xgboost_model}
    resultados = {nome: evaluate_auc_roc(modelo, X_test, y_test) for nome, modelo in modelos.items()}

    compare_models(resultados, REPORTS_DIR)
    plot_roc_curves(modelos, X_test, y_test, REPORTS_DIR)

    importancias = {nome: get_feature_importance(modelo, X_train.columns) for nome, modelo in modelos.items()}
    plot_feature_importance(importancias, REPORTS_DIR)

    # Validação cruzada: com poucos município-anos de risco nos dados
    # reais (~0.8%, 21 no total), uma única divisão treino/teste dá uma
    # AUC-ROC ruidosa -- k-fold estratificado dá uma estimativa mais
    # estável. Reusa os hiperparâmetros já escolhidos por
    # tune_random_forest/tune_xgboost (via functools.partial) em vez de
    # rebuscar a cada fold -- rebuscar por fold (nested CV completo)
    # seria mais rigoroso, mas multiplicaria o custo por n_splits sem
    # necessidade aqui.
    resultados_cv_ajustado = {
        "Random Forest": cross_validate_auc_roc(
            partial(train_random_forest, **random_forest.get_params()), datamart
        ),
        "XGBoost": cross_validate_auc_roc(partial(train_xgboost, **xgboost_model.get_params()), datamart),
    }
    report_cross_validation(resultados_cv_ajustado, REPORTS_DIR)

    # Compara, pelos mesmos folds, os hiperparâmetros ajustados contra
    # os padrão -- quantifica se a busca de hiperparâmetros (acima)
    # valeu a pena, em vez de só observar a diferença manualmente.
    resultados_cv_padrao = {
        "Random Forest": cross_validate_auc_roc(train_random_forest, datamart),
        "XGBoost": cross_validate_auc_roc(train_xgboost, datamart),
    }
    compare_tuning(resultados_cv_padrao, resultados_cv_ajustado, REPORTS_DIR)

    # Consolida os relatórios acima (compare_models, report_cross_validation,
    # compare_tuning, plot_feature_importance) num único resumo de referência.
    write_summary_report(resultados, resultados_cv_ajustado, resultados_cv_padrao, importancias, REPORTS_DIR)


def run_predict(model_filename: str = RANDOM_FOREST_FILENAME) -> Path:
    """Carrega um modelo já treinado e pontua o datamart com a probabilidade de risco.

    Só uma demonstração de inferência sobre os próprios dados de
    treino (não há dados de município-ano "novos" disponíveis fora do
    datamart) -- mas `predict.predict_risk` funciona igual sobre
    qualquer DataFrame no formato do datamart, rotulado ou não.

    Args:
        model_filename: Nome do arquivo do modelo em `models/` (ver
            `RANDOM_FOREST_FILENAME`/`XGBOOST_FILENAME`).

    Returns:
        Caminho do CSV de previsões gerado em `reports/`.
    """
    datamart = load_datamart(DATAMART_DIR)
    modelo = load_model(MODELS_DIR / model_filename)
    previsoes = predict_risk(modelo, datamart)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / PREDICTIONS_FILENAME
    previsoes.sort("probabilidade_risco", descending=True).write_csv(output_path)
    return output_path


def main() -> None:
    """Ponto de entrada do CLI (`uv run tcc-datasus <comando>`)."""
    parser = argparse.ArgumentParser(
        prog="tcc-datasus",
        description="Pipeline de predição de risco de Doença de Chagas aguda (PE/PB/AL).",
    )
    subparsers = parser.add_subparsers(dest="comando", required=True)
    subparsers.add_parser("etl", help="Baixa, converte e transforma os dados (data/processed/, data/datamart/).")
    subparsers.add_parser("train", help="Divide, ajusta hiperparâmetros, treina e avalia os modelos.")
    predict_parser = subparsers.add_parser("predict", help="Pontua o datamart com um modelo já treinado.")
    predict_parser.add_argument(
        "--model",
        default=RANDOM_FOREST_FILENAME,
        help=f"Arquivo do modelo em models/ (padrão: {RANDOM_FOREST_FILENAME}).",
    )

    args = parser.parse_args()
    if args.comando == "etl":
        run_etl()
    elif args.comando == "train":
        run_train()
    elif args.comando == "predict":
        caminho = run_predict(model_filename=args.model)
        print(f"Previsões salvas em {caminho}")
