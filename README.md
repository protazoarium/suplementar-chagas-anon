# tcc-datasus

Pipeline ETL para treinar modelos preditivos de Doença de Chagas aguda no
Nordeste, a partir de dados do SINAN/DATASUS. O escopo atual cobre as UFs
de Pernambuco (PE), Paraíba (PB) e Alagoas (AL).

> Status: as seis etapas do pipeline estão implementadas e validadas
> ponta a ponta contra dados reais do SINAN/DATASUS (2000–2025 completo,
> `FINAIS` + `PRELIM`) e do IBGE, via `run_etl.py` (download → conversão
> → transformação → datamart de risco) e `run_train.py` (divisão
> estratificada → busca de hiperparâmetros e treino, com tratamento de
> desbalanceamento → avaliação por AUC-ROC, validação cruzada e
> importância das features). Um script de inferência (`run_predict.py`)
> carrega um modelo já treinado e pontua o datamart. As três etapas
> também estão disponíveis via CLI instalado
> (`uv run tcc-datasus etl|train|predict`), e um workflow de CI
> (`.github/workflows/ci.yml`) roda `pytest`/`sqlfluff` a cada push/PR.
> Há uma suíte de testes
> automatizados no pytest (`uv run pytest`, `tests/`, 49 testes) cobrindo
> a integridade do datamart, a divisão/validação cruzada, a busca de
> hiperparâmetros, a inferência e a avaliação, e o sqlfluff está
> configurado para lintar as queries DuckDB embutidas em `transform.py`
> (`uv run sqlfluff lint .`).
> Para o histórico de decisões de projeto e problemas de dados
> encontrados durante a implementação, veja
> `reports/relatorio_2026-08-23_pipeline_etl.md` e
> `reports/relatorio_2026-08-23_tarde_risco_modelagem.md`. Para retomar
> o trabalho numa sessão nova (o que rodar, o que reconstruir), veja
> `CONTINUAR.md`.

## Pipeline

1. **Download** (`download.py`) — baixa do FTP do DATASUS os arquivos
   `.dbc` brutos do agravo SINAN-DCA (Doença de Chagas Aguda) para
   `data/raw/`. Esse agravo é distribuído apenas em arquivos nacionais
   (`CHAGBR<AA>.dbc`, sem particionamento por UF); o recorte para
   PE/PB/AL acontece na conversão. `list_available_years` distingue os
   anos consolidados (`FINAIS`) dos anos ainda sujeitos a revisão
   (`PRELIM`); `run_etl.py`/`cli.run_etl` usam os dois, cobrindo todo o
   intervalo publicado pelo DATASUS (hoje, 2000–2025) — um ano que
   aparecer nas duas fontes usa sempre a versão `FINAIS`.
2. **Conversão** (`convert.py`) — converte `.dbc` em `.dbf` (via
   `pyreaddbc`, em `data/interim/`) e então em `DataFrame` Polars, com
   parsing de datas e conversão de tipos numéricos feitos explicitamente
   a partir do schema do próprio `.dbf`, e filtro por UF (PE/PB/AL).
3. **Transformação** (`transform.py`) — harmoniza nomes de campo que
   mudaram entre eras do layout do SINAN e escreve os dados em
   `.parquet` particionado por UF e ano em
   `data/processed/uf=<UF>/ano=<ANO>/`; expõe consultas analíticas via
   DuckDB direto sobre essas partições, incluindo o cruzamento com a
   série de população do IBGE (taxas por 100 mil habitantes) e a
   agregação de casos confirmados por município-ano com features de lag
   (1 e 2 anos). É também aqui que a **variável-alvo** é rotulada:
   `label_municipio_ano_risk` marca um município-ano como "de risco"
   (1) quando teve ao menos um caso confirmado no ano, senão 0;
   `build_risk_datamart` junta esse rótulo aos lags (e, opcionalmente, à
   população/taxa por 100 mil) num datamart município-ano, persistido
   por `write_datamart_parquet` — via **PyArrow** (não o writer nativo
   do Polars usado para `data/processed/`) — em
   `data/datamart/uf=<UF>/ano=<ANO>/`, no mesmo layout Hive.
4. **Features** (`features.py`) — aplica regras epidemiológicas de
   filtragem e imputação por moda/mediana condicionada, carrega as
   partições necessárias como `LazyFrame` do Polars e constrói a matriz
   de features (demográficas, sazonais e de risco espaço-temporal) e um
   rótulo em nível de notificação (1 = Confirmado, 0 = Descartado).
   **Módulo deliberadamente independente** do datamart de risco
   município-ano da etapa 3 -- resolve uma tarefa diferente (classificar
   uma notificação individual, não prever o risco futuro de um
   município-ano) e não é consumido por `train.py`. Mantido como um
   bloco de construção completo e testado, não como código morto: uma
   eventual segunda linha de modelagem em nível de notificação partiria
   daqui.
5. **Treino** (`train.py`) — `load_datamart` carrega
   `data/datamart/`; `split_train_test` faz a divisão treino/teste via
   `sklearn.train_test_split`, **estratificada por UF e classe ao mesmo
   tempo** (chave combinada UF+rótulo, já que `stratify` só aceita um
   array); `stratified_kfold_splits` gera a mesma estratificação em `k`
   folds, para validação cruzada (ver etapa 6). As features usadas são
   `ano`, `casos_lag_1`, `casos_lag_2`, `populacao` e a UF codificada —
   `casos` do ano corrente e `taxa_por_100k` (derivada dele) ficam
   deliberadamente de fora, pois vazariam o próprio rótulo. Treina um
   `RandomForestClassifier` (`class_weight="balanced"` por padrão) e um
   `XGBClassifier` (`tree_method="hist"` **obrigatório**, algoritmo
   exato pode estourar a RAM da máquina de desenvolvimento — ver
   "Restrição de memória" abaixo — e `scale_pos_weight` calculado da
   razão negativos/positivos), ambos compensando o rótulo de risco raro
   nos dados reais deste projeto (a minoria dos município-anos, ver
   `reports/comparacao_modelos.md` para a proporção da última rodada).
   `tune_random_forest`/`tune_xgboost` buscam os melhores
   hiperparâmetros (`n_estimators`, `max_depth`, ...) via `GridSearchCV`
   sobre grades pequenas (`DEFAULT_PARAM_GRID_RF`/`DEFAULT_PARAM_GRID_XGB`
   -- deliberadamente pequenas, dado o número reduzido de município-anos
   de risco), com o mesmo `cv` estratificado por UF+rótulo e pontuado
   por `scoring="roc_auc"` (não a acurácia padrão do `GridSearchCV`);
   `class_weight`/`tree_method`/`scale_pos_weight` continuam fixos, não
   fazem parte da busca. Salva os dois modelos em `models/`;
   `get_feature_importance` extrai a importância de cada feature do
   modelo treinado.
6. **Avaliação** (`evaluate.py`) — avalia e compara os modelos
   exclusivamente pela métrica AUC-ROC (`evaluate_auc_roc`,
   `compare_models`, relatório em `reports/comparacao_modelos.md`) e
   plota as curvas ROC de todos os modelos sobrepostas no mesmo gráfico
   (`plot_roc_curves`, via `sklearn.metrics.RocCurveDisplay.from_estimator`,
   `reports/curvas_roc.png`) e a importância das features
   (`plot_feature_importance`, `reports/importancia_features.png`).
   Como o rótulo de risco é minoritário, uma única divisão treino/teste
   dá uma AUC-ROC ruidosa;
   `cross_validate_auc_roc` treina e avalia o modelo em vários folds
   estratificados (etapa 5) e `report_cross_validation` resume a
   média ± desvio padrão em `reports/validacao_cruzada.md`.
   `compare_tuning` casa, pelo nome do modelo, os resultados de
   validação cruzada com hiperparâmetros padrão e com os ajustados
   (etapa 5), mostrando lado a lado se a busca de hiperparâmetros
   realmente ajudou, em `reports/comparacao_ajuste_hiperparametros.md`.
   `write_summary_report` não recalcula nada -- só consolida o que os
   quatro relatórios acima já produzem num único documento de
   referência, `reports/resumo_avaliacao.md`, pensado para o relatório
   final do TCC.

Além das seis etapas, **`predict.py`** carrega um modelo já treinado
(`load_model`, ver `train.save_model`) e pontua município-anos com a
probabilidade prevista de risco (`predict_risk`, reaproveitando
`train.encode_features` -- a mesma codificação usada no treino, para
que a inferência nunca diverja dele). Não exige o rótulo `risco`: serve
tanto para reavaliar o próprio datamart quanto para pontuar dados
novos.

## Estrutura de diretórios

```
.github/
  workflows/ci.yml   # roda pytest + sqlfluff a cada push/PR
data/
  raw/         # .dbc originais baixados do DATASUS
  interim/     # .dbf e artefatos intermediários da conversão
  processed/   # .parquet particionado por uf=XX/ano=YYYY/ (notificação a notificação)
  datamart/    # .parquet particionado por uf=XX/ano=YYYY/ (município-ano, com o rótulo de risco)
models/        # modelos treinados serializados (.joblib)
src/
  tcc_datasus/
    download.py
    convert.py
    transform.py
    features.py
    train.py
    evaluate.py
    predict.py   # inferência: carrega um modelo treinado e pontua município-anos
    cli.py       # run_etl/run_train/run_predict + `main` do CLI instalado (uv run tcc-datasus)
scripts/
  download_convert_dca.py   # export bruto para CSV (sem filtro/imputação/Parquet) -- não é
                             # redundante com run_etl.py, serve para inspecionar o dado cru
notebooks/
  01_analise_exploratoria.ipynb   # ponto de partida: carrega o datamart e os relatórios já gerados
reports/       # relatórios e resultados de avaliação (comparacao_modelos.md, curvas_roc.png,
               # importancia_features.png, validacao_cruzada.md, comparacao_ajuste_hiperparametros.md,
               # resumo_avaliacao.md, previsoes_risco.csv)
tests/         # testes automatizados (pytest)
run_etl.py     # wrapper fino de cli.run_etl -- etapas 1-3 ponta a ponta
run_train.py   # wrapper fino de cli.run_train -- etapas 5-6 ponta a ponta
run_predict.py # wrapper fino de cli.run_predict -- pontua o datamart com um modelo treinado
.sqlfluff      # config do lint (dialeto duckdb) para as queries embutidas em transform.py
CONTINUAR.md   # como salvar/retomar o projeto entre sessões
```

`data/` e `models/` inteiros (incluindo `.dbc` e `.parquet` soltos em
qualquer lugar do repositório) são ignorados pelo git — dados e modelos
treinados nunca são versionados. Isso significa que, logo após um
`git clone`/checkout novo, essas pastas estão vazias mesmo com o pipeline
todo implementado; é preciso rebaixar os dados e retreinar (ver "Como
rodar cada etapa" abaixo).

## Ambiente

O projeto usa **uv** para gerenciar dependências e o ambiente virtual.
Não use `pip` diretamente nem crie um venv manual.

```bash
uv sync                 # instala as dependências
uv add <pacote>          # adiciona uma nova dependência
uv run <comando>         # executa um comando dentro do ambiente do projeto
```

Python 3.12 (ver `.python-version`).

## Como rodar cada etapa

Forma mais rápida de baixar e converter alguns anos do SINAN-DCA (gera
CSV em `data/interim/csv/`, restrito a PE/PB/AL):

```bash
uv run python scripts/download_convert_dca.py --anos 2020 2021 2022
```

Para rodar o pipeline completo ponta a ponta, três scripts na raiz do
projeto (cada um um wrapper fino em torno de `cli.run_etl`/
`cli.run_train`/`cli.run_predict`):

```bash
uv run python run_etl.py     # etapas 1-3: download -> conversão -> transformação
                              # -> data/processed/ e data/datamart/ (com o rótulo de risco)
uv run python run_train.py   # etapas 5-6: divisão estratificada -> busca de hiperparâmetros +
                              # treino (RF + XGBoost, com tratamento de desbalanceamento) ->
                              # avaliação (AUC-ROC, validação cruzada, comparação padrão vs.
                              # ajustado, importância das features, resumo consolidado)
                              # -> models/ e reports/
uv run python run_predict.py # carrega um modelo treinado (padrão: random_forest.joblib) e
                              # pontua o datamart -> reports/previsoes_risco.csv
```

Ou, de forma equivalente, via CLI instalado (mesmo código, `pyproject.toml`
define `tcc-datasus = "tcc_datasus:main"`):

```bash
uv run tcc-datasus etl
uv run tcc-datasus train
uv run tcc-datasus predict --model xgboost.joblib   # --model é opcional; padrão random_forest.joblib
```

`run_etl.py`/`cli.run_etl` baixam todo o intervalo publicado pelo
DATASUS: `FINAIS` (dados consolidados, hoje 2000–2022) e, para os anos
que só existem lá, `PRELIM` (2023–2025, sujeitos a revisão). Requer
acesso de rede ao FTP `ftp.datasus.gov.br` e à API do IBGE em
`apisidra.ibge.gov.br` (`fetch_ibge_population`, usada para cruzar o
datamart com a população — `populacao` entra como feature de treino;
`taxa_por_100k`, derivada de `casos` do ano corrente, não entra, pelo
mesmo motivo de `casos` na etapa 5). Leva cerca de 15 segundos nos anos
já baixados localmente; um download completo do zero demora mais.
`run_train.py` leva de 1 a 2 minutos dependendo do volume de dados, a
maior parte gasta nas duas buscas de hiperparâmetros (`GridSearchCV`).
`run_predict.py` é rápido (poucos segundos) -- só carrega o modelo já
treinado e pontua o datamart, sem treinar nada.

## Testes e lint

```bash
uv run pytest              # suíte de testes (tests/)
uv run sqlfluff lint .     # lint das queries DuckDB embutidas em transform.py
```

`.github/workflows/ci.yml` roda os dois automaticamente a cada push/PR
para `master` (`uv sync --all-groups` + `sqlfluff lint .` + `pytest`) --
só a suíte de testes, que usa dados sintéticos; `run_etl.py`/
`run_train.py` (rede + minutos de execução) não rodam em CI.

## Material suplementar anonimizado

Para a revisão às cegas do artigo, o código, os testes e os relatórios estão
disponíveis em versão anonimizada em [LINK-ANONIMIZADO-PENDENTE]. O procedimento
para gerar e aplicar o link está em `ANONIMIZACAO.md`
(`scripts/exportar_suplementar.py` e `scripts/aplicar_link_anonimo.py`).

## Restrição de memória

A máquina de desenvolvimento tem RAM limitada. Ao trabalhar com os dados:

- Prefira **Polars em modo lazy** (`pl.scan_parquet`, `LazyFrame`) em vez de
  carregar DataFrames inteiros em memória.
- Leia e processe os dados **por partição** (UF/ano), nunca o dataset
  completo de uma vez.
- Em consultas DuckDB, leia diretamente do Parquet particionado em vez de
  materializar tabelas intermediárias grandes.
