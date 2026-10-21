"""Avaliação dos modelos, exclusivamente pela métrica AUC-ROC."""

import statistics
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

try:
    _rodando_em_notebook = get_ipython() is not None  # type: ignore[name-defined]
except NameError:
    _rodando_em_notebook = False

if not _rodando_em_notebook:
    # Backend sem display (headless): precisa ser selecionado antes de
    # `matplotlib.pyplot` ser importado, já que este módulo pode rodar
    # num servidor de desenvolvimento sem X11/Wayland (ex.: run_train.py,
    # CI). Pulado dentro de um notebook/IPython -- forçar Agg ali
    # sobrescreveria o backend inline usado para exibir os gráficos (ex.:
    # notebooks/01_analise_exploratoria.ipynb), e como `tcc_datasus.cli`
    # importa este módulo, isso acontecia mesmo sem importar
    # `evaluate` diretamente (só `import tcc_datasus` já bastava).
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, RocCurveDisplay
from xgboost import XGBClassifier

from tcc_datasus.train import RANDOM_STATE_PADRAO, stratified_kfold_splits
from tcc_datasus.transform import COLUNA_RISCO

REPORT_FILENAME = "comparacao_modelos.md"
ROC_PLOT_FILENAME = "curvas_roc.png"
FEATURE_IMPORTANCE_PLOT_FILENAME = "importancia_features.png"
CV_REPORT_FILENAME = "validacao_cruzada.md"
TUNING_COMPARISON_FILENAME = "comparacao_ajuste_hiperparametros.md"
SUMMARY_REPORT_FILENAME = "resumo_avaliacao.md"
PERMUTATION_IMPORTANCE_PLOT_FILENAME = "importancia_permutacao.png"


def _mean_std(scores: list[float]) -> tuple[float, float]:
    """Média e desvio padrão de uma lista de AUC-ROC (0.0 de desvio com um só valor)."""
    return statistics.mean(scores), statistics.stdev(scores) if len(scores) > 1 else 0.0


def evaluate_auc_roc(
    model: RandomForestClassifier | XGBClassifier,
    X_test: pl.DataFrame,
    y_test: pl.Series,
) -> float:
    """Calcula a AUC-ROC do modelo no conjunto de teste.

    Métrica exclusiva de avaliação deste projeto (ver CLAUDE.md):
    acurácia, F1 e afins não são usados para decisão entre modelos, pois
    o rótulo (Confirmado/Descartado) é desbalanceado.

    Args:
        model: Modelo treinado a ser avaliado.
        X_test: Features de teste.
        y_test: Rótulos de teste.

    Returns:
        Valor da AUC-ROC.
    """
    # predict_proba retorna uma coluna por classe, na ordem de
    # model.classes_; a coluna 1 corresponde à classe positiva (rótulo 1).
    y_score = model.predict_proba(X_test.to_numpy())[:, 1]
    return roc_auc_score(y_test.to_numpy(), y_score)


def evaluate_auc_pr(
    model: RandomForestClassifier | XGBClassifier,
    X_test: pl.DataFrame,
    y_test: pl.Series,
) -> float:
    """Calcula a AUC-PR (área sob a curva precisão-revocação) do modelo no teste.

    Métrica COMPLEMENTAR, nunca usada para decidir/selecionar entre
    modelos (ver CLAUDE.md: "Avaliação exclusivamente por AUC-ROC") --
    reportada ao lado da AUC-ROC só para contextualizar o desempenho sob
    forte desbalanceamento (~2% de município-anos de risco nos dados
    reais deste projeto). Com prevalência tão baixa, a AUC-ROC pode dar
    uma leitura otimista do desempenho (Saito; Rehmsmeier, 2015, PLOS
    ONE), já que a taxa de falsos positivos (denominador = negativos,
    maioria da base) cresce devagar mesmo com muitos falsos positivos em
    termos absolutos -- a AUC-PR, cujo denominador de precisão é o total
    de positivos PREVISTOS, é mais sensível a isso.

    Args:
        model: Modelo treinado a ser avaliado.
        X_test: Features de teste.
        y_test: Rótulos de teste.

    Returns:
        Valor da AUC-PR (`sklearn.metrics.average_precision_score`).
    """
    y_score = model.predict_proba(X_test.to_numpy())[:, 1]
    return average_precision_score(y_test.to_numpy(), y_score)


def top_k_precision_recall(
    model: RandomForestClassifier | XGBClassifier,
    X_test: pl.DataFrame,
    y_test: pl.Series,
    k_valores: tuple[int, ...] = (10, 20, 50),
) -> pl.DataFrame:
    """Precisão e sensibilidade nos k município-anos de maior risco previsto.

    Métrica operacional (ver sugestão do Avaliador 2 em `sugestoes.md`):
    numa vigilância real, o que importa não é um limiar de probabilidade
    fixo, mas quantos dos município-anos priorizados (os k com maior
    probabilidade prevista) de fato eram de risco -- e que fração do
    total de município-anos de risco esses k cobrem. Complementar à
    AUC-ROC/AUC-PR, nunca usada para seleção de modelo.

    Args:
        model: Modelo treinado a ser avaliado.
        X_test: Features de teste.
        y_test: Rótulos de teste.
        k_valores: Valores de k a reportar.

    Returns:
        DataFrame com colunas `k`, `precisao` (fração dos k priorizados
        que eram de risco) e `sensibilidade` (fração de todos os
        município-anos de risco do teste cobertos pelos k priorizados).
    """
    y_score = model.predict_proba(X_test.to_numpy())[:, 1]
    y_true = y_test.to_numpy()
    ordem = np.argsort(-y_score)
    total_positivos = int(y_true.sum())

    linhas = []
    for k in k_valores:
        k_efetivo = min(k, len(y_true))
        selecionados = y_true[ordem[:k_efetivo]]
        verdadeiros_positivos = int(selecionados.sum())
        linhas.append(
            {
                "k": k,
                "precisao": verdadeiros_positivos / k_efetivo,
                "sensibilidade": verdadeiros_positivos / total_positivos if total_positivos > 0 else float("nan"),
            }
        )
    return pl.DataFrame(linhas)


def calibration_metrics(
    model: RandomForestClassifier | XGBClassifier,
    X_test: pl.DataFrame,
    y_test: pl.Series,
    n_bins: int = 10,
) -> tuple[float, pl.DataFrame]:
    """Brier score e curva de calibração das probabilidades previstas.

    Descritivo, não usado para seleção de modelo (ver CLAUDE.md). AUC-ROC
    e AUC-PR dependem só do RANKING das probabilidades previstas, não de
    seus valores absolutos -- um modelo pode discriminar bem (AUC alta)
    e ainda assim ter probabilidades mal calibradas (ex.: prever 0.9 para
    casos que na prática ocorrem com frequência 0.3). O Brier score (erro
    quadrático médio entre probabilidade prevista e rótulo observado) e a
    curva de calibração (frequência observada vs. probabilidade média
    prevista, por faixa) respondem a essa pergunta complementar.

    Args:
        model: Modelo treinado a ser avaliado.
        X_test: Features de teste.
        y_test: Rótulos de teste.
        n_bins: Número de faixas de probabilidade da curva de calibração.

    Returns:
        Tupla (brier_score, curva): `brier_score` é um único float;
        `curva` é um DataFrame com colunas `probabilidade_media_prevista`
        e `frequencia_observada`, uma linha por faixa não vazia.
    """
    y_score = model.predict_proba(X_test.to_numpy())[:, 1]
    y_true = y_test.to_numpy()
    brier = brier_score_loss(y_true, y_score)
    frequencia_observada, probabilidade_media = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    curva = pl.DataFrame(
        {
            "probabilidade_media_prevista": probabilidade_media,
            "frequencia_observada": frequencia_observada,
        }
    )
    return brier, curva


def permutation_feature_importance(
    model: RandomForestClassifier | XGBClassifier,
    X_test: pl.DataFrame,
    y_test: pl.Series,
    n_repeats: int = 30,
    random_state: int = RANDOM_STATE_PADRAO,
) -> pl.DataFrame:
    """Importância por permutação (no conjunto de TESTE), scorer = AUC-ROC.

    Complementa `train.get_feature_importance` (importância por redução
    de impureza/gain): essa importância embutida tende a favorecer
    variáveis contínuas ou de alta cardinalidade -- "ano" e "população"
    neste projeto -- mesmo quando elas não ajudam tanto a generalizar
    (Strobl et al., 2007, BMC Bioinformatics). A importância por
    permutação mede, em vez disso, quanto a AUC-ROC do modelo já treinado
    cai quando os valores de cada feature são embaralhados no conjunto de
    teste -- não é enviesada pela cardinalidade da feature, mas custa
    `n_repeats` reavaliações do modelo por feature.

    Args:
        model: Modelo treinado (RF, XGBoost, ou qualquer objeto com
            `predict_proba`).
        X_test: Features de teste (mede a importância NESTE conjunto,
            não no treino -- evita atribuir importância a padrões que o
            modelo só memorizou).
        y_test: Rótulos de teste.
        n_repeats: Repetições da permutação por feature (reduz a
            variância da estimativa).
        random_state: Semente para reprodutibilidade.

    Returns:
        DataFrame com colunas `feature`, `importancia_permutacao`
        (queda média da AUC-ROC) e `desvio_permutacao`, ordenado da
        mais para a menos importante.
    """
    resultado = permutation_importance(
        model,
        X_test.to_numpy(),
        y_test.to_numpy(),
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return pl.DataFrame(
        {
            "feature": X_test.columns,
            "importancia_permutacao": resultado.importances_mean,
            "desvio_permutacao": resultado.importances_std,
        }
    ).sort("importancia_permutacao", descending=True)


def compare_models(results: dict[str, float], report_dir: Path) -> Path:
    """Compara modelos pela AUC-ROC obtida e salva o relatório em reports/.

    Args:
        results: Mapeamento nome do modelo -> AUC-ROC obtida.
        report_dir: Diretório de saída dos relatórios (reports/).

    Returns:
        Caminho do arquivo de relatório gerado.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    ranking = sorted(results.items(), key=lambda item: item[1], reverse=True)
    melhor_nome, melhor_auc = ranking[0]

    linhas = [
        "# Comparação de modelos (AUC-ROC)",
        "",
        f"Gerado em: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "| Modelo | AUC-ROC |",
        "| --- | --- |",
        *(f"| {nome} | {auc:.4f} |" for nome, auc in ranking),
        "",
        f"Melhor modelo: **{melhor_nome}** (AUC-ROC = {melhor_auc:.4f})",
        "",
    ]

    report_path = report_dir / REPORT_FILENAME
    report_path.write_text("\n".join(linhas), encoding="utf-8")
    return report_path


def plot_roc_curves(
    models: dict[str, RandomForestClassifier | XGBClassifier],
    X_test: pl.DataFrame,
    y_test: pl.Series,
    report_dir: Path,
) -> Path:
    """Plota, sobrepostas no mesmo gráfico, as curvas ROC dos modelos avaliados.

    Usa `sklearn.metrics.RocCurveDisplay.from_estimator` (não um cálculo
    manual de `roc_curve`/`roc_auc_score`) -- a própria API do
    scikit-learn já calcula a curva e a AUC-ROC (mostrada na legenda de
    cada modelo) e desenha no mesmo eixo. Puramente uma visualização da
    mesma métrica calculada por `evaluate_auc_roc` -- não introduz
    nenhuma métrica de decisão além dela (ver CLAUDE.md, "Avaliação
    exclusivamente por AUC-ROC").

    Args:
        models: Mapeamento nome do modelo -> modelo treinado.
        X_test: Features de teste, compartilhadas entre os modelos.
        y_test: Rótulos de teste, compartilhados entre os modelos.
        report_dir: Diretório de saída dos relatórios (reports/).

    Returns:
        Caminho do arquivo de imagem gerado.
    """
    sns.set_theme(style="whitegrid")
    X_np = X_test.to_numpy()
    y_np = y_test.to_numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    nomes = list(models)
    for i, nome in enumerate(nomes):
        RocCurveDisplay.from_estimator(
            models[nome],
            X_np,
            y_np,
            name=nome,
            ax=ax,
            # Linha de referência do classificador aleatório desenhada
            # uma única vez, junto com a última curva.
            plot_chance_level=(i == len(nomes) - 1),
            chance_level_kw={"label": "Classificador aleatório (AUC-ROC = 0.5)"},
        )

    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.set_title("Curvas ROC")
    ax.legend(loc="lower right")

    report_dir.mkdir(parents=True, exist_ok=True)
    plot_path = report_dir / ROC_PLOT_FILENAME
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_feature_importance(
    importances: dict[str, pl.DataFrame],
    report_dir: Path,
    feature_col: str = "feature",
    importance_col: str = "importancia",
) -> Path:
    """Plota a importância das features de cada modelo e salva em reports/.

    Um subplot horizontal por modelo, lado a lado -- a escala de
    importância não é comparável entre Random Forest e XGBoost (ver
    `train.get_feature_importance`), então cada um usa seu próprio eixo.

    Args:
        importances: Mapeamento nome do modelo -> DataFrame de
            importância (colunas `feature_col`/`importance_col`, ver
            `train.get_feature_importance`).
        report_dir: Diretório de saída dos relatórios (reports/).
        feature_col: Coluna com o nome da feature.
        importance_col: Coluna com o valor de importância.

    Returns:
        Caminho do arquivo de imagem gerado.
    """
    sns.set_theme(style="whitegrid")
    fig, eixos = plt.subplots(1, len(importances), figsize=(6 * len(importances), 4), squeeze=False)

    for eixo, (nome, df) in zip(eixos[0], importances.items(), strict=True):
        ordenado = df.sort(importance_col)
        eixo.barh(ordenado[feature_col].to_list(), ordenado[importance_col].to_list())
        eixo.set_title(nome)
        eixo.set_xlabel("Importância")

    fig.tight_layout()
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_path = report_dir / FEATURE_IMPORTANCE_PLOT_FILENAME
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def cross_validate_auc_roc(
    treinar_modelo: Callable[[pl.DataFrame, pl.Series], RandomForestClassifier | XGBClassifier],
    df: pl.DataFrame,
    label_col: str = COLUNA_RISCO,
    uf_col: str = "uf",
    n_splits: int = 5,
    random_state: int = RANDOM_STATE_PADRAO,
) -> list[float]:
    """Valida um modelo por k-fold estratificado, retornando a AUC-ROC de cada fold.

    O rótulo de risco é raro nos dados reais deste projeto (~0.8% dos
    município-anos, 21 no total): uma única divisão treino/teste
    (`train.split_train_test`) dá uma estimativa de AUC-ROC ruidosa, já
    que o conjunto de teste fica com poucos positivos. Esta função treina
    e avalia o modelo em `n_splits` divisões diferentes
    (`train.stratified_kfold_splits`, estratificadas por UF+rótulo) e
    retorna a AUC-ROC de cada uma -- a média e o desvio padrão dessa
    lista dão uma estimativa bem mais estável do desempenho do que um
    único número (ver `report_cross_validation`).

    Args:
        treinar_modelo: Função que recebe (X_train, y_train) e retorna
            um modelo treinado -- tipicamente `train.train_random_forest`
            ou `train.train_xgboost` diretamente (ambas aceitam só esses
            dois argumentos posicionais; hiperparâmetros extras
            precisam de `functools.partial`).
        df: Datamart completo (ver `transform.build_risk_datamart`).
        label_col: Coluna do rótulo binário de risco.
        uf_col: Coluna de UF, usada (junto com o rótulo) na
            estratificação de cada fold.
        n_splits: Número de folds (ver `train.stratified_kfold_splits`
            para a restrição de tamanho mínimo por classe).
        random_state: Semente para reprodutibilidade.

    Returns:
        Lista com a AUC-ROC obtida em cada fold, na ordem em que foram
        gerados.
    """
    scores = []
    for X_train, X_test, y_train, y_test in stratified_kfold_splits(
        df, label_col=label_col, uf_col=uf_col, n_splits=n_splits, random_state=random_state
    ):
        modelo = treinar_modelo(X_train, y_train)
        scores.append(evaluate_auc_roc(modelo, X_test, y_test))
    return scores


def report_cross_validation(results: dict[str, list[float]], report_dir: Path) -> Path:
    """Resume os resultados da validação cruzada (média ± desvio da AUC-ROC) em reports/.

    Args:
        results: Mapeamento nome do modelo -> lista de AUC-ROC, uma por
            fold (ver `cross_validate_auc_roc`).
        report_dir: Diretório de saída dos relatórios (reports/).

    Returns:
        Caminho do arquivo de relatório gerado.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    resumo = {nome: _mean_std(scores) for nome, scores in results.items()}
    ranking = sorted(resumo.items(), key=lambda item: item[1][0], reverse=True)
    melhor_nome, (melhor_media, _) = ranking[0]
    n_folds = len(next(iter(results.values())))

    linhas = [
        "# Validação cruzada (AUC-ROC)",
        "",
        f"Gerado em: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Número de folds: {n_folds}",
        "",
        "| Modelo | AUC-ROC (média ± desvio) | Folds |",
        "| --- | --- | --- |",
        *(
            f"| {nome} | {media:.4f} ± {desvio:.4f} | {', '.join(f'{s:.4f}' for s in results[nome])} |"
            for nome, (media, desvio) in ranking
        ),
        "",
        f"Melhor modelo: **{melhor_nome}** (AUC-ROC média = {melhor_media:.4f})",
        "",
    ]

    report_path = report_dir / CV_REPORT_FILENAME
    report_path.write_text("\n".join(linhas), encoding="utf-8")
    return report_path


def compare_tuning(
    baseline_results: dict[str, list[float]],
    tuned_results: dict[str, list[float]],
    report_dir: Path,
) -> Path:
    """Compara, via validação cruzada, hiperparâmetros padrão e ajustados.

    Complementa `report_cross_validation` (que só ranqueia um conjunto
    de resultados de validação cruzada): aqui os resultados "padrão" e
    "ajustados" são casados pelo nome do modelo, para mostrar
    diretamente o ganho (ou perda) de médias de buscar hiperparâmetros
    (`train.tune_random_forest`/`train.tune_xgboost`) em vez de usar os
    padrões (`train.train_random_forest`/`train.train_xgboost`) --
    ambos avaliados pelos mesmos folds de `cross_validate_auc_roc`, não
    só a AUC-ROC de uma única divisão treino/teste.

    Args:
        baseline_results: Mapeamento nome do modelo -> AUC-ROC por fold
            com hiperparâmetros padrão (ver `cross_validate_auc_roc`
            passando `train.train_random_forest`/`train.train_xgboost`
            diretamente).
        tuned_results: Mesmo formato, mas com os hiperparâmetros
            escolhidos por `train.tune_random_forest`/
            `train.tune_xgboost` (tipicamente repassados via
            `functools.partial`, ver `run_train.py`).
        report_dir: Diretório de saída dos relatórios (reports/).

    Returns:
        Caminho do arquivo de relatório gerado.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    nomes = [nome for nome in baseline_results if nome in tuned_results]

    linhas = [
        "# Comparação: hiperparâmetros padrão vs. ajustados (AUC-ROC, validação cruzada)",
        "",
        f"Gerado em: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "| Modelo | Padrão (média ± desvio) | Ajustado (média ± desvio) | Δ média |",
        "| --- | --- | --- | --- |",
    ]
    for nome in nomes:
        media_padrao, desvio_padrao = _mean_std(baseline_results[nome])
        media_ajustado, desvio_ajustado = _mean_std(tuned_results[nome])
        delta = media_ajustado - media_padrao
        linhas.append(
            f"| {nome} | {media_padrao:.4f} ± {desvio_padrao:.4f} "
            f"| {media_ajustado:.4f} ± {desvio_ajustado:.4f} | {delta:+.4f} |"
        )
    linhas.append("")

    report_path = report_dir / TUNING_COMPARISON_FILENAME
    report_path.write_text("\n".join(linhas), encoding="utf-8")
    return report_path


def write_summary_report(
    single_split_results: dict[str, float],
    tuned_cv_results: dict[str, list[float]],
    baseline_cv_results: dict[str, list[float]],
    importances: dict[str, pl.DataFrame],
    report_dir: Path,
    feature_col: str = "feature",
    importance_col: str = "importancia",
    top_n_features: int = 3,
) -> Path:
    """Consolida num único relatório os resultados de todas as etapas de avaliação.

    `compare_models`, `report_cross_validation`, `compare_tuning` e
    `plot_feature_importance` foram implementados em sessões separadas
    deste projeto, cada um respondendo uma pergunta específica sobre os
    modelos e escrevendo seu próprio arquivo em `reports/`. Esta função
    não recalcula nada -- só reúne os resultados que essas funções já
    produziram (passados como argumento) num único documento de
    referência, pensado como o resumo final para o relatório do TCC, em
    vez de exigir cruzar cinco arquivos separados manualmente.

    Args:
        single_split_results: Mapeamento nome do modelo -> AUC-ROC de
            uma única divisão treino/teste (ver `evaluate_auc_roc`,
            `compare_models`).
        tuned_cv_results: Mapeamento nome do modelo -> AUC-ROC por fold,
            com os hiperparâmetros ajustados (ver `cross_validate_auc_roc`
            com `train.tune_random_forest`/`train.tune_xgboost`,
            `report_cross_validation`). Estes são os números tratados
            como o desempenho final dos modelos.
        baseline_cv_results: Mesmo formato de `tuned_cv_results`, mas
            com hiperparâmetros padrão (ver `compare_tuning`).
        importances: Mapeamento nome do modelo -> DataFrame de
            importância de features (ver `train.get_feature_importance`,
            `plot_feature_importance`).
        report_dir: Diretório de saída dos relatórios (reports/).
        feature_col: Coluna de nome da feature em `importances`.
        importance_col: Coluna de valor de importância em `importances`.
        top_n_features: Quantas features (as mais importantes) listar
            por modelo.

    Returns:
        Caminho do arquivo de relatório gerado.
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    ranking_cv = sorted(tuned_cv_results.items(), key=lambda item: statistics.mean(item[1]), reverse=True)
    melhor_nome, melhores_scores = ranking_cv[0]
    melhor_media, melhor_desvio = _mean_std(melhores_scores)

    linhas = [
        "# Resumo da avaliação dos modelos",
        "",
        f"Gerado em: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "Consolida, num único documento, o que os demais relatórios desta",
        f"etapa (`{REPORT_FILENAME}`, `{CV_REPORT_FILENAME}`, "
        f"`{TUNING_COMPARISON_FILENAME}`, `{FEATURE_IMPORTANCE_PLOT_FILENAME}`) "
        "mostram separadamente.",
        "",
        "## Desempenho final (validação cruzada, hiperparâmetros ajustados)",
        "",
        "| Modelo | AUC-ROC (média ± desvio) |",
        "| --- | --- |",
        *(f"| {nome} | {_mean_std(scores)[0]:.4f} ± {_mean_std(scores)[1]:.4f} |" for nome, scores in ranking_cv),
        "",
        f"**Melhor modelo: {melhor_nome}** (AUC-ROC média = {melhor_media:.4f} "
        f"± {melhor_desvio:.4f}). Detalhamento por fold em `{CV_REPORT_FILENAME}`.",
        "",
        "## Efeito da busca de hiperparâmetros",
        "",
        "| Modelo | Padrão (média ± desvio) | Ajustado (média ± desvio) | Δ média |",
        "| --- | --- | --- | --- |",
    ]
    for nome, scores_ajustado in tuned_cv_results.items():
        if nome not in baseline_cv_results:
            continue
        media_padrao, desvio_padrao = _mean_std(baseline_cv_results[nome])
        media_ajustado, desvio_ajustado = _mean_std(scores_ajustado)
        delta = media_ajustado - media_padrao
        linhas.append(
            f"| {nome} | {media_padrao:.4f} ± {desvio_padrao:.4f} "
            f"| {media_ajustado:.4f} ± {desvio_ajustado:.4f} | {delta:+.4f} |"
        )

    linhas += [
        "",
        f"Detalhamento completo em `{TUNING_COMPARISON_FILENAME}`.",
        "",
        "## AUC-ROC de uma única divisão treino/teste (referência)",
        "",
        "| Modelo | AUC-ROC |",
        "| --- | --- |",
        *(f"| {nome} | {auc:.4f} |" for nome, auc in single_split_results.items()),
        "",
        "Só como referência -- com o rótulo de risco raro nos dados reais",
        "deste projeto (poucos município-anos de risco), uma única divisão",
        "dá uma estimativa mais ruidosa que a validação cruzada acima (ver",
        f"seção anterior e `{CV_REPORT_FILENAME}`). Detalhamento em `{REPORT_FILENAME}`.",
        "",
        "## Features mais importantes",
        "",
    ]
    for nome, df in importances.items():
        top = df.sort(importance_col, descending=True).head(top_n_features)
        descricao = ", ".join(
            f"{feature} ({importancia:.3f})"
            for feature, importancia in zip(top[feature_col].to_list(), top[importance_col].to_list(), strict=True)
        )
        linhas.append(f"- **{nome}**: {descricao}")

    linhas += [
        "",
        f"Gráfico completo em `{FEATURE_IMPORTANCE_PLOT_FILENAME}`; curvas ROC em `{ROC_PLOT_FILENAME}`.",
        "",
    ]

    report_path = report_dir / SUMMARY_REPORT_FILENAME
    report_path.write_text("\n".join(linhas), encoding="utf-8")
    return report_path


def report_nested_cv(
    nao_aninhado: dict[str, list[float]],
    aninhado: dict[str, list[float]],
    report_dir: Path,
    filename: str = "validacao_cruzada_aninhada.md",
) -> Path:
    """Compara a validação cruzada "ajustada" não aninhada com a aninhada.

    A versão não aninhada usada no resumo original (`compare_tuning`,
    `resultados_cv_ajustado` em `cli.run_train`) reusa, em TODOS os folds
    de uma nova validação cruzada sobre o datamart inteiro, os
    hiperparâmetros escolhidos uma única vez via `tune_random_forest`/
    `tune_xgboost` sobre a divisão treino/teste original -- como essa
    divisão original e os novos folds vêm do mesmo datamart, parte dos
    município-anos usados para ESCOLHER os hiperparâmetros acaba caindo
    no conjunto de TESTE de algum fold novo (otimismo, consistente com a
    crítica do Avaliador 2 e com a queda observada de ~0.91 na CV para
    ~0.86-0.87 na divisão de teste). A versão aninhada (`aninhado`,
    produzida chamando `train.cross_validate_auc_roc` com
    `train.tune_random_forest`/`train.tune_xgboost` -- que já fazem sua
    própria busca interna -- como `treinar_modelo`) refaz a busca de
    hiperparâmetros a cada fold externo, usando só os dados de treino
    DAQUELE fold -- sem esse vazamento.

    Args:
        nao_aninhado: Mapeamento nome do modelo -> AUC-ROC por fold, sem
            busca de hiperparâmetros aninhada (ver `compare_tuning`).
        aninhado: Mesmo formato, com busca de hiperparâmetros aninhada.
        report_dir: Diretório de saída dos relatórios (reports/).
        filename: Nome do arquivo de relatório.

    Returns:
        Caminho do arquivo de relatório gerado.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    nomes = [nome for nome in nao_aninhado if nome in aninhado]

    linhas = [
        "# Validação cruzada aninhada vs. não aninhada (AUC-ROC)",
        "",
        f"Gerado em: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "| Modelo | Não aninhada (média ± desvio) | Aninhada (média ± desvio) | Δ média |",
        "| --- | --- | --- | --- |",
    ]
    for nome in nomes:
        media_na, desvio_na = _mean_std(nao_aninhado[nome])
        media_an, desvio_an = _mean_std(aninhado[nome])
        delta = media_an - media_na
        linhas.append(
            f"| {nome} | {media_na:.4f} ± {desvio_na:.4f} | {media_an:.4f} ± {desvio_an:.4f} | {delta:+.4f} |"
        )
    linhas += [
        "",
        "A versão aninhada é a estimativa honesta de desempenho (sem o",
        "otimismo da busca de hiperparâmetros não aninhada); a diferença",
        "para a versão não aninhada explica, ao menos em parte, a queda",
        "observada entre a AUC-ROC de validação cruzada (~0.91 na versão",
        "original do artigo) e a AUC-ROC da divisão única de teste",
        "(~0.86-0.87).",
        "",
    ]

    report_path = report_dir / filename
    report_path.write_text("\n".join(linhas), encoding="utf-8")
    return report_path


def report_temporal_validation(
    resultados: dict[str, list[tuple[int, float, float]]],
    report_dir: Path,
    filename: str = "validacao_temporal.md",
) -> Path:
    """Relata a AUC-ROC/AUC-PR por split de validação temporal (origem deslizante).

    Args:
        resultados: Mapeamento "nome do modelo (com/sem ano)" -> lista de
            tuplas `(ano_teste, auc_roc, auc_pr)`, uma por split (ver
            `train.temporal_splits`).
        report_dir: Diretório de saída dos relatórios (reports/).
        filename: Nome do arquivo de relatório.

    Returns:
        Caminho do arquivo de relatório gerado.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    linhas = [
        "# Validação temporal (origem deslizante): treino em ano <= t, teste em t+1",
        "",
        f"Gerado em: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "Anos sem nenhum município-ano de risco no teste (AUC-ROC indefinida)",
        "são omitidos (ver `train.temporal_splits`).",
        "",
    ]
    for nome, splits in resultados.items():
        aucs_roc = [auc for _, auc, _ in splits]
        aucs_pr = [pr for _, _, pr in splits]
        media_roc, desvio_roc = _mean_std(aucs_roc)
        media_pr, desvio_pr = _mean_std(aucs_pr)
        linhas += [
            f"## {nome}",
            "",
            f"AUC-ROC média ± desvio: {media_roc:.4f} ± {desvio_roc:.4f}  ",
            f"AUC-PR média ± desvio: {media_pr:.4f} ± {desvio_pr:.4f}",
            "",
            "| Ano de teste | AUC-ROC | AUC-PR |",
            "| --- | --- | --- |",
            *(f"| {ano} | {auc:.4f} | {pr:.4f} |" for ano, auc, pr in splits),
            "",
        ]

    report_path = report_dir / filename
    report_path.write_text("\n".join(linhas), encoding="utf-8")
    return report_path


def report_baselines(
    resultados: dict[str, dict[str, float]],
    report_dir: Path,
    filename: str = "baselines.md",
) -> Path:
    """Relata AUC-ROC/AUC-PR dos modelos e dos baselines de referência, lado a lado.

    Args:
        resultados: Mapeamento nome do modelo/baseline -> dict com (ao
            menos) as chaves `auc_roc` e `auc_pr`. Espera-se que inclua
            tanto RF/XGBoost quanto `RegraCasosAnoAnterior`,
            "Regressão logística" e "Regressão logística (ano + UF)".
        report_dir: Diretório de saída dos relatórios (reports/).
        filename: Nome do arquivo de relatório.

    Returns:
        Caminho do arquivo de relatório gerado.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    ranking = sorted(resultados.items(), key=lambda item: item[1]["auc_roc"], reverse=True)
    linhas = [
        "# Modelos vs. baselines de referência",
        "",
        f"Gerado em: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "| Modelo/baseline | AUC-ROC | AUC-PR |",
        "| --- | --- | --- |",
        *(f"| {nome} | {m['auc_roc']:.4f} | {m['auc_pr']:.4f} |" for nome, m in ranking),
        "",
    ]
    report_path = report_dir / filename
    report_path.write_text("\n".join(linhas), encoding="utf-8")
    return report_path


def plot_permutation_importance(
    importancias: dict[str, pl.DataFrame],
    report_dir: Path,
) -> Path:
    """Plota a importância por permutação de cada modelo e salva em reports/.

    Mesmo layout de `plot_feature_importance`, mas para o resultado de
    `permutation_feature_importance` (barras de erro = desvio entre
    repetições da permutação).

    Args:
        importancias: Mapeamento nome do modelo -> DataFrame de
            `permutation_feature_importance`.
        report_dir: Diretório de saída dos relatórios (reports/).

    Returns:
        Caminho do arquivo de imagem gerado.
    """
    sns.set_theme(style="whitegrid")
    fig, eixos = plt.subplots(1, len(importancias), figsize=(6 * len(importancias), 4), squeeze=False)

    for eixo, (nome, df) in zip(eixos[0], importancias.items(), strict=True):
        ordenado = df.sort("importancia_permutacao")
        eixo.barh(
            ordenado["feature"].to_list(),
            ordenado["importancia_permutacao"].to_list(),
            xerr=ordenado["desvio_permutacao"].to_list(),
        )
        eixo.set_title(nome)
        eixo.set_xlabel("Queda média da AUC-ROC (importância por permutação)")

    fig.tight_layout()
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_path = report_dir / PERMUTATION_IMPORTANCE_PLOT_FILENAME
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path
