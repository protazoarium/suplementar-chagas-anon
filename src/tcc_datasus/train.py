"""Treino dos modelos preditivos (Random Forest e XGBoost)."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from tcc_datasus.features import MAPA_UF
from tcc_datasus.transform import COLUNA_RISCO

# Semente padrão para reprodutibilidade; pode ser sobrescrita via **params.
RANDOM_STATE_PADRAO = 42

# Grades de busca deliberadamente pequenas: com só 259 município-anos de
# risco no total nos dados reais deste projeto (~2,14%, ver
# `stratified_kfold_splits`), uma grade grande arrisca escolher
# hiperparâmetros que só se ajustam ao ruído de uma amostra tão pequena
# ("overfitting" da própria busca), sem ganho real de generalização.
DEFAULT_PARAM_GRID_RF: dict[str, list[Any]] = {
    "n_estimators": [200, 500],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
DEFAULT_PARAM_GRID_XGB: dict[str, list[Any]] = {
    "n_estimators": [100, 300],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1, 0.3],
}

COLUNA_UF_COD = "UF_COD"

# Colunas do datamart (ver `transform.build_risk_datamart`) usadas como
# features. `casos` (contagem de casos confirmados do ano corrente) e
# `taxa_por_100k` (derivada de `casos`) são deliberadamente excluídas: o
# rótulo `risco` é definido a partir de `casos` (ver
# `transform.label_municipio_ano_risk`), então usá-las como feature
# vazaria o próprio rótulo para o modelo. Só sobram os lags (informação
# de anos anteriores, sem vazamento) e a população.
COLUNAS_FEATURES_DATAMART = ("ano", "casos_lag_1", "casos_lag_2", "populacao")


def load_datamart(datamart_dir: Path) -> pl.DataFrame:
    """Carrega o datamart analítico particionado (ver `transform.write_datamart_parquet`).

    Args:
        datamart_dir: Diretório raiz do datamart em Parquet (ex.:
            data/datamart/), no layout Hive uf=<UF>/ano=<ANO>/ gerado por
            `transform.write_datamart_parquet`.

    Returns:
        DataFrame com uma linha por município-ano (colunas de partição
        uf/ano incluídas).
    """
    return pl.read_parquet(datamart_dir / "**" / "*.parquet", hive_partitioning=True)


def encode_features(df: pl.DataFrame, uf_col: str = "uf") -> pl.DataFrame:
    """Codifica numericamente as features do datamart (UF -> `UF_COD`, seleção de colunas).

    Única fonte de verdade para "como um datamart vira X" -- usada tanto
    no treino (`split_train_test`/`stratified_kfold_splits`, via
    `_encode_features_and_stratification_key`) quanto na inferência
    (`predict.predict_risk`), para garantir a mesma codificação nos dois
    casos. Não precisa da coluna de rótulo: funciona também sobre
    município-anos ainda não rotulados (ex.: para pontuar dados novos).

    Args:
        df: Datamart (ver `transform.build_risk_datamart`), com `uf_col`
            e as colunas presentes de `COLUNAS_FEATURES_DATAMART`.
        uf_col: Coluna de UF, codificada numericamente (ver
            `features.MAPA_UF`).

    Returns:
        DataFrame só com as features já codificadas numericamente.
    """
    df = df.with_columns(
        pl.col(uf_col).replace_strict(MAPA_UF, default=None, return_dtype=pl.Int64).alias(COLUNA_UF_COD)
    )
    colunas_features = [c for c in COLUNAS_FEATURES_DATAMART if c in df.columns] + [COLUNA_UF_COD]
    return df.select(colunas_features)


def _encode_features_and_stratification_key(
    df: pl.DataFrame,
    label_col: str,
    uf_col: str,
) -> tuple[pl.DataFrame, pl.Series, np.ndarray]:
    """Codifica as features e monta a chave de estratificação UF+rótulo.

    Lógica compartilhada por `split_train_test` e `stratified_kfold_splits`
    -- os dois precisam da mesma codificação de features (`encode_features`)
    e da mesma chave combinada (o `stratify` do scikit-learn só aceita um
    único array; ver `split_train_test` para o porquê da combinação).

    Returns:
        Tupla (X, y, chave_estratificacao): features codificadas, rótulo
        e a chave `"<uf>_<rótulo>"` usada para estratificar.
    """
    X = encode_features(df, uf_col)
    y = df[label_col]
    chave_estratificacao = (df[uf_col].cast(pl.Utf8) + "_" + df[label_col].cast(pl.Utf8)).to_numpy()
    return X, y, chave_estratificacao


def split_train_test(
    df: pl.DataFrame,
    label_col: str = COLUNA_RISCO,
    uf_col: str = "uf",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE_PADRAO,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    """Divide o datamart em treino/teste, estratificado por UF e classe.

    O parâmetro `stratify` do scikit-learn aceita só um único array, então
    UF e rótulo são combinados numa chave auxiliar (ex.: "PE_1") antes da
    divisão -- isso preserva, simultaneamente em treino e teste, tanto a
    proporção de cada UF quanto a proporção de município-anos de risco
    dentro de cada UF.

    Com o rótulo de risco raro nos dados reais deste projeto (~2,14% dos
    município-anos, 259 no total), uma única divisão como esta dá uma
    estimativa de AUC-ROC ruidosa -- o conjunto de teste fica com poucos
    positivos. Para uma estimativa mais estável, ver
    `stratified_kfold_splits`.

    Args:
        df: Datamart completo (ver `transform.build_risk_datamart`), com
            `uf_col`, `label_col` e as colunas presentes de
            `COLUNAS_FEATURES_DATAMART`.
        label_col: Coluna do rótulo binário de risco.
        uf_col: Coluna de UF, usada (junto com o rótulo) na
            estratificação e codificada numericamente como feature (ver
            `features.MAPA_UF`).
        test_size: Fração dos dados reservada para teste.
        random_state: Semente para reprodutibilidade.

    Returns:
        Tupla (X_train, X_test, y_train, y_test): features já
        codificadas numericamente e o rótulo binário de risco.
    """
    X, y, chave_estratificacao = _encode_features_and_stratification_key(df, label_col, uf_col)
    colunas_features = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(),
        y.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=chave_estratificacao,
    )
    return (
        pl.DataFrame(X_train, schema=colunas_features),
        pl.DataFrame(X_test, schema=colunas_features),
        pl.Series(label_col, y_train),
        pl.Series(label_col, y_test),
    )


def stratified_kfold_splits(
    df: pl.DataFrame,
    label_col: str = COLUNA_RISCO,
    uf_col: str = "uf",
    n_splits: int = 5,
    random_state: int = RANDOM_STATE_PADRAO,
) -> Iterator[tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]]:
    """Gera os folds de validação cruzada, estratificados por UF e classe.

    Mesma chave de estratificação combinada UF+rótulo de
    `split_train_test`, mas repetida em `n_splits` divisões diferentes
    (via `sklearn.model_selection.StratifiedKFold`) em vez de uma única
    divisão treino/teste -- dá uma estimativa de AUC-ROC bem mais estável
    quando a classe positiva é rara, como o rótulo de risco deste
    projeto (ver `split_train_test`).

    Args:
        df: Datamart completo (ver `transform.build_risk_datamart`), com
            `uf_col`, `label_col` e as colunas presentes de
            `COLUNAS_FEATURES_DATAMART`.
        label_col: Coluna do rótulo binário de risco.
        uf_col: Coluna de UF, usada (junto com o rótulo) na
            estratificação de cada fold.
        n_splits: Número de folds. Cada classe presente na chave
            UF+rótulo precisa ter pelo menos `n_splits` membros
            (`StratifiedKFold` levanta erro caso contrário) -- nos dados
            reais deste projeto, a menor classe de risco tem 36
            município-anos (AL, contra 79 em PB e 144 em PE), então o
            padrão (5) é seguro; aumentar demais pode quebrar.
        random_state: Semente para reprodutibilidade.

    Yields:
        Tuplas (X_train, X_test, y_train, y_test), uma por fold, no
        mesmo formato de `split_train_test`.
    """
    X, y, chave_estratificacao = _encode_features_and_stratification_key(df, label_col, uf_col)
    colunas_features = X.columns
    X_np, y_np = X.to_numpy(), y.to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for idx_treino, idx_teste in skf.split(X_np, chave_estratificacao):
        yield (
            pl.DataFrame(X_np[idx_treino], schema=colunas_features),
            pl.DataFrame(X_np[idx_teste], schema=colunas_features),
            pl.Series(label_col, y_np[idx_treino]),
            pl.Series(label_col, y_np[idx_teste]),
        )


def temporal_splits(
    df: pl.DataFrame,
    label_col: str = COLUNA_RISCO,
    uf_col: str = "uf",
    ano_col: str = "ano",
    remover_ano: bool = False,
    anos_sem_notificacao: tuple[int, ...] = (2007, 2008, 2009),
) -> Iterator[tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series, int]]:
    """Gera splits de origem deslizante: treina em ano <= t, testa em t+1.

    Resposta direta à preocupação do Avaliador 2 em `sugestoes.md`: a
    validação cruzada estratificada (`stratified_kfold_splits`) embaralha
    anos aleatoriamente entre treino e teste, então o modelo pode "ver"
    município-anos de um ano também presente no teste (mesmo município-ano
    não, mas o mesmo ANO em municípios diferentes) -- isso não está
    disponível no uso real (prever o ano seguinte a partir dos anos já
    vistos). Esta função simula esse uso real: para cada ano `t` com dados
    suficientes (ver abaixo), treina com todos os município-anos de ano
    <= t e testa nos município-anos de ano == t + 1.

    Anos em `anos_sem_notificacao` (2007-2009 nos dados reais deste
    projeto -- lacuna de publicação do SINAN-DCA para PE/PB/AL nesse
    período, não ausência real de casos) são pulados como ano de TESTE:
    `y_test` ficaria inteiramente 0 (todo município-ano sem risco), o que
    torna a AUC-ROC indefinida (precisa das duas classes presentes). Esses
    anos continuam no conjunto de TREINO dos splits seguintes normalmente
    (`casos`/`casos_lag_*` já valem 0 ali, refletindo a lacuna real).

    Args:
        df: Datamart completo (ver `transform.build_risk_datamart`), com
            `uf_col`, `ano_col`, `label_col` e as colunas presentes de
            `COLUNAS_FEATURES_DATAMART`.
        label_col: Coluna do rótulo binário de risco.
        uf_col: Coluna de UF, codificada numericamente como feature (ver
            `features.MAPA_UF`) -- NÃO usada para vazamento temporal,
            só como feature, igual a `split_train_test`.
        ano_col: Coluna de ano.
        remover_ano: Se True, remove a coluna `ano` das features (`UF_COD`
            continua presente) -- usado para comparar o desempenho com e
            sem essa feature sob o mesmo esquema de validação temporal
            (ver discussão do Avaliador 2 sobre o atributo "ano" dominar
            a importância por concentrar informação de tendência/ano
            específico que não generaliza para anos fora do treino).
        anos_sem_notificacao: Anos a pular como ano de TESTE (ver acima).

    Yields:
        Tuplas (X_train, X_test, y_train, y_test, ano_teste): features já
        codificadas numericamente (mesmo formato de `split_train_test`) e
        o ano usado como teste naquele split.
    """
    anos_disponiveis = sorted(df[ano_col].unique().to_list())
    for ano_teste in anos_disponiveis:
        ano_limite_treino = ano_teste - 1
        if ano_limite_treino not in anos_disponiveis or ano_teste in anos_sem_notificacao:
            continue

        treino_df = df.filter(pl.col(ano_col) <= ano_limite_treino)
        teste_df = df.filter(pl.col(ano_col) == ano_teste)
        if treino_df[label_col].n_unique() < 2 or teste_df[label_col].n_unique() < 2:
            # Sem as duas classes presentes (treino ou teste), AUC-ROC
            # não é definida para este split -- pula silenciosamente
            # (ex.: os primeiros anos da série, com poucos ou nenhum
            # município-ano de risco ainda).
            continue

        X_train = encode_features(treino_df, uf_col)
        X_test = encode_features(teste_df, uf_col)
        if remover_ano:
            X_train = X_train.drop(ano_col)
            X_test = X_test.drop(ano_col)

        yield X_train, X_test, treino_df[label_col], teste_df[label_col], ano_teste


def train_random_forest(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    **params: Any,
) -> RandomForestClassifier:
    """Treina um classificador Random Forest.

    Args:
        X_train: Features de treino (já codificadas numericamente, sem
            valores ausentes -- ver `features.build_features`).
        y_train: Rótulos de treino.
        **params: Hiperparâmetros repassados ao RandomForestClassifier;
            `random_state=42` e `class_weight="balanced"` são usados por
            padrão caso não informados. `class_weight="balanced"`
            pondera cada classe pelo inverso da sua frequência em
            `y_train`: o rótulo de risco é raro nos dados reais deste
            projeto (~2,14% dos município-anos, ver
            `transform.label_municipio_ano_risk`), e sem essa
            ponderação a árvore tende a nunca prever a classe
            minoritária.

    Returns:
        Modelo Random Forest treinado.
    """
    params.setdefault("random_state", RANDOM_STATE_PADRAO)
    params.setdefault("class_weight", "balanced")
    modelo = RandomForestClassifier(**params)
    modelo.fit(X_train.to_numpy(), y_train.to_numpy())
    return modelo


def train_xgboost(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    **params: Any,
) -> XGBClassifier:
    """Treina um classificador XGBoost.

    Args:
        X_train: Features de treino (já codificadas numericamente -- ver
            `features.build_features`). Ao contrário do Random Forest, o
            XGBoost lida nativamente com valores ausentes (NaN).
        y_train: Rótulos de treino.
        **params: Hiperparâmetros repassados ao XGBClassifier;
            `random_state=42`, `tree_method="hist"` e `scale_pos_weight`
            são usados por padrão caso não informados.
            `tree_method="hist"` é obrigatório neste projeto (ver
            CLAUDE.md, "Restrição de memória"): o algoritmo exato
            ("exact", padrão do XGBoost) enumera todos os pontos de
            corte possíveis e pode estourar a RAM da máquina de
            desenvolvimento; "hist" usa histogramas de features e tem
            custo de memória previsível. `scale_pos_weight` (razão
            negativos/positivos em `y_train`) compensa o mesmo
            desbalanceamento do rótulo de risco tratado em
            `train_random_forest` (`class_weight="balanced"`), mas pelo
            mecanismo próprio do XGBoost -- não é calculado (fica em 1,
            o padrão do XGBoost) se `y_train` não tiver nenhum exemplo
            positivo.

    Returns:
        Modelo XGBoost treinado.
    """
    params.setdefault("random_state", RANDOM_STATE_PADRAO)
    params.setdefault("tree_method", "hist")
    if "scale_pos_weight" not in params:
        n_positivos = int((y_train == 1).sum())
        n_negativos = int((y_train == 0).sum())
        if n_positivos > 0:
            params["scale_pos_weight"] = n_negativos / n_positivos
    modelo = XGBClassifier(**params)
    modelo.fit(X_train.to_numpy(), y_train.to_numpy())
    return modelo


class RegraCasosAnoAnterior:
    """Baseline sem aprendizado: pontua pelo número de casos confirmados no ano anterior.

    Serve de referência mínima para contextualizar a AUC-ROC dos modelos
    treinados (ver sugestão do Avaliador 2 em `sugestoes.md`: "uma regra
    baseada em casos no ano anterior"). Implementa a mesma interface
    `predict_proba` de um classificador do scikit-learn (só a coluna 1,
    da classe positiva, é usada por `evaluate.evaluate_auc_roc` e
    congêneres) para encaixar nas mesmas funções de avaliação dos modelos
    "de verdade", sem precisar de um `fit` -- a regra não tem parâmetro
    algum a estimar.

    Como AUC-ROC/AUC-PR dependem só do ranking relativo das pontuações
    (não da escala), pontuar diretamente por `casos_lag_1` (em vez de uma
    regra binária 0/1) dá uma curva ROC/PR não degenerada, mais informativa
    que comparar só um único ponto de operação.
    """

    def __init__(self, coluna: str = "casos_lag_1") -> None:
        self.coluna = coluna
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegraCasosAnoAnterior":
        """Não-operação: a regra não tem parâmetros a ajustar a partir dos dados."""
        return self

    def predict_proba(self, X: pl.DataFrame | np.ndarray) -> np.ndarray:
        """Pontua cada linha por `casos_lag_1` (ou a coluna informada), normalizado para [0, 1].

        Args:
            X: Features já codificadas (ver `encode_features`), como
                DataFrame (usa o nome da coluna) ou array numpy (usa a
                posição de `self.coluna` em `COLUNAS_FEATURES_DATAMART`
                + `COLUNA_UF_COD`, a mesma ordem produzida por
                `encode_features`).

        Returns:
            Array (n_linhas, 2): coluna 0 = 1 - score, coluna 1 = score
            normalizado pelo valor máximo observado em X (evita pontuação
            > 1; quando todas as linhas têm o mesmo valor, todas recebem
            score 0).
        """
        if isinstance(X, pl.DataFrame):
            bruto = X[self.coluna].to_numpy().astype(float)
        else:
            idx = list(COLUNAS_FEATURES_DATAMART).index(self.coluna)
            bruto = X[:, idx].astype(float)
        # Município-anos no primeiro ano da série não têm `casos_lag_1`
        # observado (ver `transform.aggregate_municipio_ano_lags`); nulo
        # aqui significa "sem notificação conhecida no ano anterior", o
        # mesmo sinal que um valor 0 -- sem este fill_null, um único NaN
        # faz `np.ndarray.max()` retornar NaN (não ignora nulos por
        # padrão) e degenerava a pontuação inteira para zero.
        bruto = np.nan_to_num(bruto, nan=0.0)
        maximo = bruto.max()
        score_positivo = bruto / maximo if maximo > 0 else np.zeros_like(bruto)
        return np.column_stack([1 - score_positivo, score_positivo])


def train_logistic_regression(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    colunas: list[str] | None = None,
    **params: Any,
) -> Any:
    """Treina uma regressão logística como baseline linear.

    Usada tanto com todas as features (baseline "modelo simples" pedido
    pelo Avaliador 2) quanto, via `colunas=["ano", "UF_COD"]`, como o
    baseline "só ano e UF" -- para medir quanto da AUC-ROC dos modelos
    de árvore vem só dessas duas variáveis (que dominam a importância
    por impureza/gain, ver `get_feature_importance`).

    Valores ausentes são imputados por 0 (`SimpleImputer`) e as features
    são padronizadas (`StandardScaler`) antes da regressão logística,
    dentro de um `sklearn.pipeline.Pipeline`. A imputação por 0 é
    necessária porque `casos_lag_2` (e `casos_lag_1`) ficam nulos nos
    primeiros anos da série município-ano, sem histórico anterior
    disponível (ver `transform.aggregate_municipio_ano_lags`) -- RF e
    XGBoost aceitam esses nulos nativamente, mas a regressão logística
    do scikit-learn não. A padronização é necessária porque, ao
    contrário de árvores, a regressão logística é sensível à escala das
    features (ex.: `populacao`, em milhares, vs. `casos_lag_1`, de 0 a
    poucas unidades).

    Args:
        X_train: Features de treino (ver `split_train_test`).
        y_train: Rótulos de treino.
        colunas: Subconjunto de colunas de `X_train` a usar; `None` usa
            todas.
        **params: Hiperparâmetros repassados a `LogisticRegression`;
            `random_state=42` e `class_weight="balanced"` são usados por
            padrão (mesma razão de `train_random_forest`: o rótulo de
            risco é raro).

    Returns:
        Pipeline treinado (`StandardScaler` + `LogisticRegression`); tem
        `predict_proba`, compatível com `evaluate.evaluate_auc_roc`.
    """
    params.setdefault("random_state", RANDOM_STATE_PADRAO)
    params.setdefault("class_weight", "balanced")
    X = X_train.select(colunas) if colunas is not None else X_train
    modelo = make_pipeline(SimpleImputer(strategy="constant", fill_value=0), StandardScaler(), LogisticRegression(**params))
    modelo.fit(X.to_numpy(), y_train.to_numpy())
    if colunas is not None:
        # `evaluate_auc_roc` chama `model.predict_proba(X_test.to_numpy())`
        # com todas as colunas de X_test; como este modelo só conhece um
        # subconjunto, a seleção de colunas precisa acontecer de novo na
        # hora de pontuar -- `_RestringirColunas` (abaixo) embrulha o
        # pipeline para isso.
        return _RestringirColunas(modelo, colunas, X_train.columns)
    return modelo


class _RestringirColunas:
    """Embrulha um modelo treinado só num subconjunto de colunas de X_train.

    `evaluate.evaluate_auc_roc` e as demais funções de avaliação sempre
    chamam `model.predict_proba(X_test.to_numpy())` com TODAS as colunas
    de `encode_features` (para funcionar igual com RF/XGBoost/regressão
    logística completos) -- este wrapper seleciona, na hora de pontuar,
    só as colunas que o modelo interno de fato usa (ver
    `train_logistic_regression` com `colunas=["ano", "UF_COD"]`).
    """

    def __init__(self, modelo: Any, colunas: list[str], colunas_completas: list[str]) -> None:
        self.modelo = modelo
        self.indices = [colunas_completas.index(c) for c in colunas]
        self.classes_ = modelo.classes_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.modelo.predict_proba(X[:, self.indices])


def _stratified_folds_from_features(X_train: pl.DataFrame, y_train: pl.Series, n_splits: int, random_state: int):
    """Monta os folds (índices) de `X_train` estratificados por UF+rótulo.

    Mesma ideia de `stratified_kfold_splits`, mas operando sobre
    features já codificadas (`X_train` já tem a coluna `UF_COD`, ver
    `_encode_features_and_stratification_key`) em vez do datamart bruto
    -- usado para passar um `cv` pronto ao `GridSearchCV` em
    `tune_random_forest`/`tune_xgboost`, que já recebem X_train/y_train
    (a mesma convenção de `train_random_forest`/`train_xgboost`).
    """
    chave_estratificacao = (X_train[COLUNA_UF_COD].cast(pl.Utf8) + "_" + y_train.cast(pl.Utf8)).to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(X_train.to_numpy(), chave_estratificacao))


def tune_random_forest(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    param_grid: dict[str, list[Any]] | None = None,
    n_splits: int = 5,
    scoring: str = "roc_auc",
    random_state: int = RANDOM_STATE_PADRAO,
) -> RandomForestClassifier:
    """Busca os melhores hiperparâmetros de Random Forest via grid search.

    Usa `sklearn.model_selection.GridSearchCV` com `cv` estratificado por
    UF+rótulo (mesma chave de `split_train_test`/`stratified_kfold_splits`,
    ver `_stratified_folds_from_features`) em vez do `cv` numérico padrão
    do scikit-learn, para não perder a rara classe de risco de nenhuma UF
    num fold. `scoring="roc_auc"` por padrão -- este projeto avalia
    exclusivamente por AUC-ROC (ver CLAUDE.md), então a busca de
    hiperparâmetros usa a mesma métrica, não acurácia (o padrão do
    `GridSearchCV`). `class_weight="balanced"` (ver `train_random_forest`)
    é mantido fixo, não faz parte da busca.

    Com `refit=True` (padrão do `GridSearchCV`), o modelo retornado já
    está treinado com os melhores hiperparâmetros sobre todo `X_train`/
    `y_train` -- pode ser usado diretamente no lugar de
    `train_random_forest` (ex.: em `evaluate.evaluate_auc_roc`,
    `get_feature_importance`, `save_model`).

    Args:
        X_train: Features de treino (ver `split_train_test`).
        y_train: Rótulos de treino.
        param_grid: Grade de hiperparâmetros a testar; `None` usa
            `DEFAULT_PARAM_GRID_RF` (deliberadamente pequena -- ver o
            comentário na constante).
        n_splits: Número de folds internos da busca (mesma restrição de
            tamanho mínimo por classe de `stratified_kfold_splits`).
        scoring: Métrica usada para escolher os melhores hiperparâmetros.
        random_state: Semente para reprodutibilidade (dos folds e do
            `RandomForestClassifier`).

    Returns:
        Modelo Random Forest com os melhores hiperparâmetros
        encontrados, já treinado em todo `X_train`/`y_train`.
    """
    cv = _stratified_folds_from_features(X_train, y_train, n_splits, random_state)
    modelo_base = RandomForestClassifier(random_state=random_state, class_weight="balanced")
    busca = GridSearchCV(modelo_base, param_grid or DEFAULT_PARAM_GRID_RF, scoring=scoring, cv=cv)
    busca.fit(X_train.to_numpy(), y_train.to_numpy())
    return busca.best_estimator_


def tune_xgboost(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    param_grid: dict[str, list[Any]] | None = None,
    n_splits: int = 5,
    scoring: str = "roc_auc",
    random_state: int = RANDOM_STATE_PADRAO,
) -> XGBClassifier:
    """Busca os melhores hiperparâmetros de XGBoost via grid search.

    Mesma lógica de `tune_random_forest` (grade pequena, `cv`
    estratificado por UF+rótulo, `scoring="roc_auc"` por padrão), mas
    para `XGBClassifier`. `tree_method="hist"` (obrigatório, ver
    `train_xgboost`) e `scale_pos_weight` (calculado da razão
    negativos/positivos em `y_train`, mesmo cálculo de `train_xgboost`)
    são mantidos fixos, não fazem parte da busca.

    Args:
        X_train: Features de treino (ver `split_train_test`).
        y_train: Rótulos de treino.
        param_grid: Grade de hiperparâmetros a testar; `None` usa
            `DEFAULT_PARAM_GRID_XGB`.
        n_splits: Número de folds internos da busca.
        scoring: Métrica usada para escolher os melhores hiperparâmetros.
        random_state: Semente para reprodutibilidade.

    Returns:
        Modelo XGBoost com os melhores hiperparâmetros encontrados, já
        treinado em todo `X_train`/`y_train`.
    """
    n_positivos = int((y_train == 1).sum())
    n_negativos = int((y_train == 0).sum())
    scale_pos_weight = n_negativos / n_positivos if n_positivos > 0 else 1.0

    cv = _stratified_folds_from_features(X_train, y_train, n_splits, random_state)
    modelo_base = XGBClassifier(random_state=random_state, tree_method="hist", scale_pos_weight=scale_pos_weight)
    busca = GridSearchCV(modelo_base, param_grid or DEFAULT_PARAM_GRID_XGB, scoring=scoring, cv=cv)
    busca.fit(X_train.to_numpy(), y_train.to_numpy())
    return busca.best_estimator_


def save_model(model: RandomForestClassifier | XGBClassifier, path: Path) -> None:
    """Persiste um modelo treinado em disco.

    Args:
        model: Modelo treinado (Random Forest ou XGBoost).
        path: Caminho de destino do arquivo serializado.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def get_feature_importance(
    model: RandomForestClassifier | XGBClassifier,
    feature_names: list[str],
) -> pl.DataFrame:
    """Extrai a importância de cada feature do modelo treinado.

    Usa `model.feature_importances_`, disponível tanto no
    `RandomForestClassifier` (importância por redução média de
    impureza) quanto no `XGBClassifier` (importância por "gain"
    acumulado, exposta pela API scikit-learn do XGBoost) -- os dois não
    são diretamente comparáveis em escala absoluta entre si, só o
    ranking relativo dentro de cada modelo é significativo.

    Args:
        model: Modelo treinado (Random Forest ou XGBoost).
        feature_names: Nomes das colunas de X_train/X_test, na mesma
            ordem usada no treino (ver `split_train_test`; tipicamente
            `X_train.columns`).

    Returns:
        DataFrame com colunas `feature` e `importancia`, ordenado da
        mais para a menos importante.
    """
    return pl.DataFrame({"feature": feature_names, "importancia": model.feature_importances_}).sort(
        "importancia", descending=True
    )
