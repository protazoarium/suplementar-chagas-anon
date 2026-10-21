"""Inferência: carrega um modelo treinado e pontua município-anos com o risco previsto."""

from pathlib import Path

import joblib
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from tcc_datasus.train import encode_features

COLUNA_PROBABILIDADE_PADRAO = "probabilidade_risco"
COLUNA_PREVISAO_PADRAO = "risco_previsto"
LIMIAR_PREVISAO_PADRAO = 0.5


def load_model(path: Path) -> RandomForestClassifier | XGBClassifier:
    """Carrega um modelo treinado, salvo por `train.save_model`.

    Args:
        path: Caminho do arquivo `.joblib` do modelo (ex.: models/random_forest.joblib).

    Returns:
        Modelo treinado, pronto para `predict_risk`.
    """
    return joblib.load(path)


def predict_risk(
    model: RandomForestClassifier | XGBClassifier,
    df: pl.DataFrame,
    uf_col: str = "uf",
    limiar: float = LIMIAR_PREVISAO_PADRAO,
    probabilidade_col: str = COLUNA_PROBABILIDADE_PADRAO,
    previsao_col: str = COLUNA_PREVISAO_PADRAO,
) -> pl.DataFrame:
    """Pontua município-anos com a probabilidade de risco prevista pelo modelo.

    Codifica as features exatamente como no treino (`train.encode_features`
    -- a mesma função usada por `train.split_train_test`), então não exige
    que `df` já tenha o rótulo `risco`: serve tanto para reavaliar
    município-anos já rotulados quanto para pontuar dados novos (ex.: um
    ano ainda não incluído no datamart de treino).

    Args:
        model: Modelo treinado (ver `load_model`).
        df: Datamart (mesmas colunas de `transform.build_risk_datamart`).
        uf_col: Coluna de UF.
        limiar: Probabilidade mínima para o rótulo previsto ser 1.
        probabilidade_col: Nome da coluna de probabilidade a adicionar.
        previsao_col: Nome da coluna de rótulo previsto a adicionar.

    Returns:
        `df` original com as colunas `probabilidade_col` e `previsao_col`
        adicionadas (as demais colunas de `df`, incluindo identificadoras
        como `municipio`/`uf`/`ano`, são preservadas).
    """
    X = encode_features(df, uf_col)
    probabilidades = model.predict_proba(X.to_numpy())[:, 1]
    coluna_probabilidade = pl.Series(probabilidade_col, probabilidades)
    return df.with_columns(
        coluna_probabilidade,
        (coluna_probabilidade >= limiar).cast(pl.Int8).alias(previsao_col),
    )
