# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projeto

Pipeline ETL para treinar modelos preditivos de Doença de Chagas aguda no Nordeste (TCC), usando dados do SINAN/DATASUS restritos às UFs de Pernambuco, Paraíba e Alagoas.

Cada etapa do pipeline vive em um módulo próprio dentro de `src/tcc_datasus/`, hoje contendo apenas assinaturas de função com docstrings e type hints (sem implementação):

1. **Download** (`download.py`) — arquivos `.dbc` brutos do SINAN/DATASUS (via `datasus-dbc`) → `data/raw/`.
2. **Conversão** (`convert.py`) — `.dbc` → `.dbf` (`data/interim/`) → DataFrame (via `pyreaddbc`/`dbfread`).
3. **Transformação** (`transform.py`) — escrita em `.parquet` particionado por UF e ano em `data/processed/uf=<UF>/ano=<ANO>/`, e consultas analíticas via DuckDB direto sobre essas partições.
4. **Features** (`features.py`) — carregamento lazy das partições e engenharia de features com Polars.
5. **Treino** (`train.py`) — Random Forest e XGBoost.
6. **Avaliação** (`evaluate.py`) — exclusivamente por AUC-ROC (não usar acurácia, F1, etc. como métrica de decisão).

Ao implementar qualquer etapa, mantenha essa ordem e os formatos intermediários (`.dbc` → `.dbf` → Parquet particionado) em vez de introduzir formatos alternativos. `data/` inteiro (e qualquer `.dbc`/`.parquet` avulso) é ignorado pelo git — nunca versione dados.

## Ambiente e comandos

- Gerenciador de pacotes/ambiente é o **uv**. Nunca use `pip` diretamente nem crie um venv manual.
  - Instalar dependência: `uv add <pacote>`
  - Executar qualquer comando/script: `uv run <comando>`
  - Ponto de entrada do CLI: `uv run tcc-datasus`
- Python 3.12 (fixado em `.python-version`).
- Build via `uv_build` (ver `[build-system]` em `pyproject.toml`).

## Restrição de memória

A máquina de desenvolvimento tem RAM limitada:
- Prefira **Polars em modo lazy** (`pl.scan_parquet`, `.lazy()`) em vez de carregar DataFrames inteiros em memória.
- Leia e processe os dados **por partição** (UF/ano), nunca o dataset completo de uma vez.
- Ao consultar via DuckDB, favoreça queries que leiam diretamente do Parquet particionado em vez de materializar tabelas intermediárias grandes.

## Dependências principais

- `datasus-dbc`, `pyreaddbc`, `dbfread` — leitura dos formatos nativos `.dbc`/`.dbf` do DATASUS.
- `duckdb`, `polars`, `pandas`, `pyarrow` — carregamento, consulta e transformação dos dados.
- `scikit-learn` — Random Forest e avaliação (AUC-ROC).
- `xgboost` — modelo de gradient boosting.
- `matplotlib`, `seaborn` — visualização.
- `jupyterlab` — notebooks exploratórios (`uv run jupyter lab`).

Não há framework de testes, linter ou CI configurado ainda.
