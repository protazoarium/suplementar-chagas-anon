"""Testes da divisão estratificada treino/teste e do treino dos modelos."""

import polars as pl
import pytest

from tcc_datasus.train import (
    COLUNA_UF_COD,
    COLUNAS_FEATURES_DATAMART,
    encode_features,
    get_feature_importance,
    split_train_test,
    stratified_kfold_splits,
    train_random_forest,
    train_xgboost,
    tune_random_forest,
    tune_xgboost,
)

# 4 UFs fictícias x 50 município-anos cada, risco raro (5%) para
# exercitar a estratificação combinada UF+classe num dataset pequeno,
# igual ao datamart real (label desbalanceado).
N_POR_UF = 50
DATAMART_SINTETICO = pl.DataFrame(
    {
        "uf": [uf for uf in ("PE", "PB", "AL") for _ in range(N_POR_UF)],
        "ano": list(range(2000, 2000 + N_POR_UF)) * 3,
        "casos": [1 if i % 20 == 0 else 0 for i in range(N_POR_UF)] * 3,
        "casos_lag_1": [0] * (N_POR_UF * 3),
        "casos_lag_2": [0] * (N_POR_UF * 3),
        "populacao": [1_000_000] * (N_POR_UF * 3),
        "risco": [1 if i % 20 == 0 else 0 for i in range(N_POR_UF)] * 3,
    }
)

# Base maior, com pelo menos 5 município-anos de risco por UF (10% de
# 100), para exercitar stratified_kfold_splits com n_splits=5 --
# StratifiedKFold exige que toda classe da chave de estratificação
# (UF+rótulo) tenha pelo menos n_splits membros.
N_POR_UF_CV = 100
DATAMART_SINTETICO_CV = pl.DataFrame(
    {
        "uf": [uf for uf in ("PE", "PB", "AL") for _ in range(N_POR_UF_CV)],
        "ano": list(range(2000, 2000 + N_POR_UF_CV)) * 3,
        "casos": [1 if i % 10 == 0 else 0 for i in range(N_POR_UF_CV)] * 3,
        "casos_lag_1": [0] * (N_POR_UF_CV * 3),
        "casos_lag_2": [0] * (N_POR_UF_CV * 3),
        "populacao": [1_000_000] * (N_POR_UF_CV * 3),
        "risco": [1 if i % 10 == 0 else 0 for i in range(N_POR_UF_CV)] * 3,
    }
)


def test_encode_features_nao_exige_rotulo():
    sem_risco = DATAMART_SINTETICO.drop("risco")
    X = encode_features(sem_risco)
    assert set(X.columns) == {*COLUNAS_FEATURES_DATAMART, COLUNA_UF_COD}
    assert len(X) == len(sem_risco)


def test_encode_features_mesmo_resultado_que_split_train_test():
    X_train, X_test, _, _ = split_train_test(DATAMART_SINTETICO)
    X_completo = encode_features(DATAMART_SINTETICO)
    assert len(X_train) + len(X_test) == len(X_completo)
    assert set(X_train.columns) == set(X_completo.columns)


def test_split_train_test_preserva_colunas_de_features_sem_vazamento():
    X_train, X_test, y_train, y_test = split_train_test(DATAMART_SINTETICO)
    for coluna_vazamento in ("casos", "risco", "taxa_por_100k"):
        assert coluna_vazamento not in X_train.columns
        assert coluna_vazamento not in X_test.columns
    for coluna in COLUNAS_FEATURES_DATAMART:
        assert coluna in X_train.columns


def test_split_train_test_preserva_tamanho_total():
    X_train, X_test, y_train, y_test = split_train_test(DATAMART_SINTETICO, test_size=0.2)
    assert len(X_train) + len(X_test) == len(DATAMART_SINTETICO)
    assert len(y_train) + len(y_test) == len(DATAMART_SINTETICO)


def test_split_train_test_estratifica_proporcao_de_risco():
    _, _, y_train, y_test = split_train_test(DATAMART_SINTETICO, test_size=0.2)
    proporcao_total = DATAMART_SINTETICO["risco"].mean()
    assert abs(y_train.mean() - proporcao_total) < 0.05
    assert abs(y_test.mean() - proporcao_total) < 0.05


def test_split_train_test_estratifica_proporcao_por_uf():
    X_train, X_test, _, _ = split_train_test(DATAMART_SINTETICO, test_size=0.2)
    # UF_COD é a codificação numérica de MAPA_UF (features.MAPA_UF); com
    # as 3 UFs igualmente representadas na base sintética, cada uma deve
    # continuar igualmente representada nos dois splits.
    contagem_treino = X_train["UF_COD"].value_counts().sort("UF_COD")["count"]
    contagem_teste = X_test["UF_COD"].value_counts().sort("UF_COD")["count"]
    assert contagem_treino.to_list() == [len(X_train) // 3] * 3
    assert contagem_teste.to_list() == [len(X_test) // 3] * 3


def test_split_train_test_reprodutivel_com_mesma_semente():
    a = split_train_test(DATAMART_SINTETICO, random_state=1)
    b = split_train_test(DATAMART_SINTETICO, random_state=1)
    assert a[0].equals(b[0])
    assert a[2].to_list() == b[2].to_list()


def test_train_random_forest_treina_e_prediz():
    X_train, X_test, y_train, y_test = split_train_test(DATAMART_SINTETICO)
    modelo = train_random_forest(X_train, y_train)
    predicoes = modelo.predict(X_test.to_numpy())
    assert len(predicoes) == len(X_test)
    assert set(predicoes.tolist()) <= {0, 1}


def test_train_xgboost_usa_tree_method_hist_por_padrao():
    X_train, _, y_train, _ = split_train_test(DATAMART_SINTETICO)
    modelo = train_xgboost(X_train, y_train)
    assert modelo.get_params()["tree_method"] == "hist"


def test_train_xgboost_tree_method_explicito_nao_e_sobrescrito():
    X_train, _, y_train, _ = split_train_test(DATAMART_SINTETICO)
    modelo = train_xgboost(X_train, y_train, tree_method="approx")
    assert modelo.get_params()["tree_method"] == "approx"


def test_train_random_forest_usa_class_weight_balanced_por_padrao():
    X_train, _, y_train, _ = split_train_test(DATAMART_SINTETICO)
    modelo = train_random_forest(X_train, y_train)
    assert modelo.get_params()["class_weight"] == "balanced"


def test_train_random_forest_class_weight_explicito_nao_e_sobrescrito():
    X_train, _, y_train, _ = split_train_test(DATAMART_SINTETICO)
    modelo = train_random_forest(X_train, y_train, class_weight=None)
    assert modelo.get_params()["class_weight"] is None


def test_train_xgboost_calcula_scale_pos_weight_pela_razao_neg_pos():
    X_train, _, y_train, _ = split_train_test(DATAMART_SINTETICO)
    modelo = train_xgboost(X_train, y_train)
    n_negativos = int((y_train == 0).sum())
    n_positivos = int((y_train == 1).sum())
    assert modelo.get_params()["scale_pos_weight"] == pytest.approx(n_negativos / n_positivos)


def test_train_xgboost_scale_pos_weight_explicito_nao_e_sobrescrito():
    X_train, _, y_train, _ = split_train_test(DATAMART_SINTETICO)
    modelo = train_xgboost(X_train, y_train, scale_pos_weight=1.0)
    assert modelo.get_params()["scale_pos_weight"] == 1.0


def test_train_xgboost_sem_positivos_mantem_scale_pos_weight_padrao():
    X_train = pl.DataFrame({"x": [0, 1, 2, 3]})
    y_train = pl.Series("risco", [0, 0, 0, 0])
    modelo = train_xgboost(X_train, y_train)
    assert modelo.get_params()["scale_pos_weight"] is None


def test_get_feature_importance_ordenada_e_soma_compativel():
    X_train, _, y_train, _ = split_train_test(DATAMART_SINTETICO)
    modelo = train_random_forest(X_train, y_train)
    importancias = get_feature_importance(modelo, X_train.columns)

    assert set(importancias["feature"].to_list()) == set(X_train.columns)
    valores = importancias["importancia"].to_list()
    assert valores == sorted(valores, reverse=True)
    assert sum(valores) == pytest.approx(1.0)


def test_stratified_kfold_splits_gera_n_splits_folds():
    folds = list(stratified_kfold_splits(DATAMART_SINTETICO_CV, n_splits=5))
    assert len(folds) == 5


def test_stratified_kfold_splits_cada_linha_aparece_no_teste_exatamente_uma_vez():
    folds = list(stratified_kfold_splits(DATAMART_SINTETICO_CV, n_splits=5))
    tamanhos_teste = [len(X_test) for _, X_test, _, _ in folds]
    assert sum(tamanhos_teste) == len(DATAMART_SINTETICO_CV)


def test_stratified_kfold_splits_preserva_proporcao_de_risco_em_cada_fold():
    proporcao_total = DATAMART_SINTETICO_CV["risco"].mean()
    for _, _, _, y_test in stratified_kfold_splits(DATAMART_SINTETICO_CV, n_splits=5):
        assert abs(y_test.mean() - proporcao_total) < 0.05


def test_stratified_kfold_splits_sem_vazamento_de_casos_ou_risco():
    for X_train, X_test, _, _ in stratified_kfold_splits(DATAMART_SINTETICO_CV, n_splits=5):
        for coluna_vazamento in ("casos", "risco", "taxa_por_100k"):
            assert coluna_vazamento not in X_train.columns
            assert coluna_vazamento not in X_test.columns


def test_tune_random_forest_escolhe_dentro_da_grade_e_mantem_class_weight():
    X_train, X_test, y_train, _ = split_train_test(DATAMART_SINTETICO_CV)
    modelo = tune_random_forest(X_train, y_train, param_grid={"n_estimators": [10, 20]}, n_splits=2)
    predicoes = modelo.predict(X_test.to_numpy())

    assert len(predicoes) == len(X_test)
    assert modelo.get_params()["n_estimators"] in (10, 20)
    assert modelo.get_params()["class_weight"] == "balanced"


def test_tune_xgboost_escolhe_dentro_da_grade_e_calcula_scale_pos_weight():
    X_train, X_test, y_train, _ = split_train_test(DATAMART_SINTETICO_CV)
    modelo = tune_xgboost(
        X_train, y_train, param_grid={"n_estimators": [10, 20], "max_depth": [2, 3]}, n_splits=2
    )
    predicoes = modelo.predict(X_test.to_numpy())

    assert len(predicoes) == len(X_test)
    assert modelo.get_params()["n_estimators"] in (10, 20)
    assert modelo.get_params()["max_depth"] in (2, 3)
    assert modelo.get_params()["tree_method"] == "hist"
    n_positivos = int((y_train == 1).sum())
    n_negativos = int((y_train == 0).sum())
    assert modelo.get_params()["scale_pos_weight"] == pytest.approx(n_negativos / n_positivos)
