"""Escrita em Parquet particionado e consultas analíticas via DuckDB."""

import json
import urllib.request
from importlib import resources
from pathlib import Path

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.dataset as pa_dataset

from tcc_datasus.convert import UF_IBGE_CODE
from tcc_datasus.features import CODIGO_CONFIRMADO, COLUNAS_COM_ALIAS

# Mapeamento inverso: código IBGE -> sigla de UF.
IBGE_CODE_TO_UF = {codigo: uf for uf, codigo in UF_IBGE_CODE.items()}

PARTITION_FILENAME = "part-0.parquet"
DATAMART_BASENAME_TEMPLATE = "part-{i}.parquet"

# Um município-ano é rotulado "de risco" (1) quando teve ao menos este
# número de casos confirmados de Doença de Chagas aguda no ano; abaixo
# disso (tipicamente 0), é rotulado "sem risco" (0). Ver
# `label_municipio_ano_risk`.
LIMIAR_RISCO_PADRAO = 1
COLUNA_RISCO = "risco"

# SIDRA (IBGE): tabela 6579 = "População residente estimada", variável
# 9324, nível geográfico n3 = Unidade da Federação (para a série
# agregada por UF) ou n6 = Município (para a série municipal, ver
# `fetch_ibge_population_municipal`).
SIDRA_URL = "https://apisidra.ibge.gov.br/values/t/6579/n3/{ufs}/v/9324/p/all"
SIDRA_URL_MUNICIPAL = "https://apisidra.ibge.gov.br/values/t/6579/n6/{municipios}/v/9324/p/all"

# Lista oficial (IBGE, API de localidades) dos 510 municípios de PE/PB/AL,
# com o código IBGE de 7 dígitos -- congelada em arquivo versionado em vez
# de buscada em tempo de execução, já que a divisão político-administrativa
# do Brasil não muda com frequência e este projeto não deve depender de
# rede para montar a grade município-ano (ver `load_municipios_oficiais`).
MUNICIPIOS_RESOURCE = "resources/municipios_pe_pb_al.csv"


def load_municipios_oficiais() -> pl.DataFrame:
    """Carrega a lista oficial de municípios de PE/PB/AL (código IBGE de 7 dígitos).

    Usada para montar a grade município-ano a partir do universo completo
    de municípios (ver `aggregate_municipio_ano_lags`), em vez de apenas
    dos códigos que aparecem em `data/processed/` -- essencial para não
    omitir municípios sem nenhuma notificação em 26 anos.

    Returns:
        DataFrame com colunas `codigo_ibge` (7 dígitos, str), `uf`, `nome`.
    """
    caminho = resources.files("tcc_datasus") / MUNICIPIOS_RESOURCE
    return pl.read_csv(caminho.read_bytes(), schema_overrides={"codigo_ibge": pl.Utf8})


def harmonize_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Renomeia campos cujo nome mudou entre eras do layout do SINAN-DCA.

    Por volta de 2007-2010 vários campos foram renomeados (ex.:
    CON_CLASSI -> CLASSI_FIN, ver `features.COLUNAS_COM_ALIAS`). Sem essa
    normalização, escrever partições de anos de eras diferentes produz
    arquivos Parquet com schemas incompatíveis, e consultas via DuckDB que
    atravessem esses anos falham (a coluna canônica simplesmente não
    existe nos arquivos do layout antigo).

    Args:
        df: DataFrame de um ano/partição, com nomes de coluna daquela era.

    Returns:
        DataFrame com as colunas renomeadas para o nome canônico (o
        primeiro de cada tupla em `COLUNAS_COM_ALIAS`) quando aplicável.
    """
    renomeios = {
        legado: canonico
        for canonico, *legados in COLUNAS_COM_ALIAS
        for legado in legados
        if legado in df.columns and canonico not in df.columns
    }
    return df.rename(renomeios) if renomeios else df


def prepare_partition_columns(
    df: pl.DataFrame,
    uf_col_origem: str = "SG_UF_NOT",
    ano_col_origem: str = "NU_ANO",
    uf_col: str = "uf",
    ano_col: str = "ano",
) -> pl.DataFrame:
    """Prepara um DataFrame para `write_partitioned_parquet`.

    Deriva colunas limpas de UF (sigla) e ano (inteiro) e harmoniza os
    campos cujo nome mudou entre eras do SINAN-DCA (ver `harmonize_schema`).

    O SINAN representa a UF de notificação ora como código IBGE ("26"),
    ora como sigla ("PE"), dependendo do ano do arquivo (ver
    `convert.UF_IBGE_CODE`). Esta função normaliza os dois casos para a
    sigla e converte o ano para inteiro.

    Args:
        df: DataFrame de origem (tipicamente já filtrado/imputado).
        uf_col_origem: Coluna com a UF bruta (código IBGE ou sigla).
        ano_col_origem: Coluna com o ano bruto (texto).
        uf_col: Nome da coluna de UF normalizada a criar.
        ano_col: Nome da coluna de ano normalizado a criar.

    Returns:
        DataFrame com as colunas `uf_col`/`ano_col` adicionadas e os
        campos de nome variável renomeados para a forma canônica.
    """
    df = harmonize_schema(df)
    return df.with_columns(
        pl.col(uf_col_origem).replace(IBGE_CODE_TO_UF).alias(uf_col),
        pl.col(ano_col_origem).cast(pl.Int64).alias(ano_col),
    )


def write_partitioned_parquet(
    df: pl.DataFrame,
    output_dir: Path,
    uf_col: str = "uf",
    ano_col: str = "ano",
) -> None:
    """Escreve um DataFrame em .parquet particionado por UF e ano.

    O layout de saída segue o padrão uf=<UF>/ano=<ANO>/ dentro de
    output_dir (data/processed/). Cada combinação UF/ano vira um único
    arquivo Parquet, sobrescrito a cada chamada (idempotente): reprocessar
    o mesmo ano/UF substitui a partição em vez de duplicar dados.

    Args:
        df: DataFrame com os dados a serem escritos. Precisa conter as
            colunas `uf_col` e `ano_col` já normalizadas (ver
            `prepare_partition_columns`).
        output_dir: Diretório raiz de saída (data/processed/).
        uf_col: Nome da coluna usada para particionar por UF.
        ano_col: Nome da coluna usada para particionar por ano.
    """
    if df.is_empty():
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    particoes = df.partition_by([uf_col, ano_col], as_dict=True, include_key=True)

    for (uf, ano), grupo in particoes.items():
        destino = output_dir / f"{uf_col}={uf}" / f"{ano_col}={ano}"
        destino.mkdir(parents=True, exist_ok=True)
        grupo.write_parquet(destino / PARTITION_FILENAME)


def _read_processed_view(con: duckdb.DuckDBPyConnection, processed_dir: Path) -> None:
    padrao_glob = str(processed_dir / "**" / "*.parquet")
    # union_by_name: mesmo com `harmonize_schema` aplicado, partições
    # ainda podem diferir em colunas menos essenciais entre eras; isso
    # evita que uma consulta que não toque nessas colunas falhe por causa
    # delas (viram NULL nos arquivos onde não existem).
    con.read_parquet(padrao_glob, hive_partitioning=True, union_by_name=True).create_view("processed")


def query_processed(query: str, processed_dir: Path) -> pl.DataFrame:
    """Executa uma consulta analítica via DuckDB sobre os dados particionados.

    Os arquivos em `processed_dir` (layout uf=<UF>/ano=<ANO>/*.parquet) são
    expostos como a view `processed`, lida diretamente do Parquet
    particionado via DuckDB (sem materializar uma tabela intermediária em
    memória) -- `query` deve referenciar essa view, ex.:
    "SELECT uf, count(*) FROM processed GROUP BY uf".

    Args:
        query: Consulta SQL a ser executada pelo DuckDB.
        processed_dir: Diretório raiz dos dados em Parquet (data/processed/).

    Returns:
        Resultado da consulta como DataFrame Polars.
    """
    con = duckdb.connect(":memory:")
    try:
        _read_processed_view(con, processed_dir)
        return con.execute(query).pl()
    finally:
        con.close()


def fetch_ibge_population(
    ufs: tuple[str, ...] = ("PE", "PB", "AL"),
    ano_inicio: int = 2000,
    ano_fim: int = 2025,
) -> pl.DataFrame:
    """Busca a série de população residente estimada do IBGE por UF/ano.

    Usa a API pública do SIDRA (tabela 6579, variável 9324) em nível de
    Unidade da Federação. Essa tabela não cobre anos de censo/contagem
    (2000, 2007, 2010, 2022) nem 2023 -- esses anos são preenchidos por
    interpolação linear a partir dos anos vizinhos disponíveis (ou, nas
    bordas da série, repetindo o valor mais próximo), e marcados na
    coluna `estimado`.

    Args:
        ufs: Siglas de UF a buscar.
        ano_inicio: Primeiro ano da série (inclusive).
        ano_fim: Último ano da série (inclusive).

    Returns:
        DataFrame com colunas uf, ano, populacao, estimado (True para
        anos preenchidos por interpolação/repetição, não vindos direto do
        IBGE).
    """
    codigos_ibge = [UF_IBGE_CODE[uf] for uf in ufs]
    url = SIDRA_URL.format(ufs=",".join(codigos_ibge))
    with urllib.request.urlopen(url, timeout=30) as resposta:
        registros = json.loads(resposta.read())[1:]  # 1a linha é cabeçalho

    bruto = pl.DataFrame(
        {
            "uf": [IBGE_CODE_TO_UF[r["D1C"]] for r in registros],
            "ano": [int(r["D3C"]) for r in registros],
            "populacao": [int(r["V"]) for r in registros],
        }
    )

    grade = pl.DataFrame({"ano": list(range(ano_inicio, ano_fim + 1))}).join(
        pl.DataFrame({"uf": list(ufs)}), how="cross"
    )
    serie = grade.join(bruto, on=["uf", "ano"], how="left").sort(["uf", "ano"])
    serie = serie.with_columns(pl.col("populacao").is_null().alias("estimado"))
    # Interpola anos entre dois pontos conhecidos; nas bordas da série
    # (sem vizinho de um dos lados), repete o valor conhecido mais próximo.
    serie = serie.with_columns(pl.col("populacao").interpolate().over("uf"))
    serie = serie.with_columns(pl.col("populacao").fill_null(strategy="forward").over("uf"))
    serie = serie.with_columns(pl.col("populacao").fill_null(strategy="backward").over("uf"))

    return serie.with_columns(pl.col("populacao").round().cast(pl.Int64))


def fetch_ibge_population_municipal(
    municipios_oficiais: pl.DataFrame | None = None,
    ano_inicio: int = 2000,
    ano_fim: int = 2025,
    tamanho_lote: int = 100,
) -> pl.DataFrame:
    """Busca a série de população residente estimada do IBGE por município/ano.

    Mesma tabela SIDRA 6579/variável 9324 de `fetch_ibge_population`, mas
    no nível geográfico n6 (Município) em vez de n3 (UF) -- ao contrário
    do que o artigo original afirmava, essa tabela TEM granularidade
    municipal (ver Seção 5.2 revisada). Usada para dar ao modelo a
    população do município em vez da UF como feature, já que a UF
    combinada com ano/UF_COD funciona quase como um identificador
    UF-ano, sem poder de discriminar municípios entre si.

    A API SIDRA aceita múltiplos códigos de município na mesma URL
    (separados por vírgula); para evitar uma URL/resposta grande demais
    com os 510 municípios de uma vez, a busca é feita em lotes de
    `tamanho_lote` códigos.

    Args:
        municipios_oficiais: Lista oficial de municípios (ver
            `load_municipios_oficiais`); `None` carrega a lista padrão
            (PE/PB/AL) do arquivo versionado.
        ano_inicio: Primeiro ano da série (inclusive).
        ano_fim: Último ano da série (inclusive).
        tamanho_lote: Quantos códigos de município pedir por requisição.

    Returns:
        DataFrame com colunas `municipio` (código IBGE de 7 dígitos),
        `uf`, `ano`, `populacao`, `estimado` (True para anos preenchidos
        por interpolação/repetição, não vindos direto do IBGE) -- mesma
        semântica de `fetch_ibge_population`, por município.
    """
    municipios_oficiais = municipios_oficiais if municipios_oficiais is not None else load_municipios_oficiais()
    codigos = municipios_oficiais["codigo_ibge"].to_list()

    registros: list[dict] = []
    for inicio in range(0, len(codigos), tamanho_lote):
        lote = codigos[inicio : inicio + tamanho_lote]
        url = SIDRA_URL_MUNICIPAL.format(municipios=",".join(lote))
        with urllib.request.urlopen(url, timeout=60) as resposta:
            registros.extend(json.loads(resposta.read())[1:])  # 1a linha é cabeçalho

    bruto = pl.DataFrame(
        {
            "municipio": [r["D1C"] for r in registros],
            "ano": [int(r["D3C"]) for r in registros],
            "populacao": [int(r["V"]) for r in registros],
        }
    )

    grade = (
        pl.DataFrame({"ano": list(range(ano_inicio, ano_fim + 1))})
        .join(municipios_oficiais.select("codigo_ibge", "uf").rename({"codigo_ibge": "municipio"}), how="cross")
    )
    serie = grade.join(bruto, on=["municipio", "ano"], how="left").sort(["municipio", "ano"])
    serie = serie.with_columns(pl.col("populacao").is_null().alias("estimado"))
    serie = serie.with_columns(pl.col("populacao").interpolate().over("municipio"))
    serie = serie.with_columns(pl.col("populacao").fill_null(strategy="forward").over("municipio"))
    serie = serie.with_columns(pl.col("populacao").fill_null(strategy="backward").over("municipio"))

    return serie.with_columns(pl.col("populacao").round().cast(pl.Int64))


def calculate_rates_per_100k(
    processed_dir: Path,
    populacao: pl.DataFrame,
    uf_col: str = "uf",
    ano_col: str = "ano",
    habitantes_base: int = 100_000,
) -> pl.DataFrame:
    """Cruza os casos confirmados (dados limpos) com a população do IBGE.

    Conta, via DuckDB, os casos confirmados (CLASSI_FIN = "1") por UF/ano
    em `processed_dir`, junta com `populacao` (ver `fetch_ibge_population`)
    e calcula a taxa de incidência por 100 mil habitantes. Combinações
    UF/ano sem nenhum caso aparecem com 0 casos (LEFT JOIN a partir da
    população), não ficam ausentes do resultado.

    Args:
        processed_dir: Diretório raiz dos dados em Parquet (data/processed/),
            já com o schema harmonizado (ver `prepare_partition_columns`).
        populacao: DataFrame de população por UF/ano (colunas `uf_col`,
            `ano_col`, "populacao"), tipicamente de `fetch_ibge_population`.
        uf_col: Nome da coluna de UF, tanto em `processed_dir` quanto em
            `populacao`.
        ano_col: Nome da coluna de ano, tanto em `processed_dir` quanto em
            `populacao`.
        habitantes_base: Base da taxa (100 mil, por padrão).

    Returns:
        DataFrame com uma linha por UF/ano: casos confirmados, população
        e taxa por `habitantes_base` habitantes.
    """
    con = duckdb.connect(":memory:")
    try:
        _read_processed_view(con, processed_dir)
        con.register("populacao", populacao)
        query = f"""
            WITH casos_confirmados AS (
                SELECT
                    {uf_col},
                    {ano_col},
                    count(*) AS casos_confirmados
                FROM processed
                WHERE CLASSI_FIN = '{CODIGO_CONFIRMADO}'
                GROUP BY {uf_col}, {ano_col}
            )

            SELECT
                p.{uf_col},
                p.{ano_col},
                p.populacao,
                coalesce(c.casos_confirmados, 0) AS casos_confirmados,
                coalesce(c.casos_confirmados, 0)
                * ?::DOUBLE
                / p.populacao AS taxa_por_100k
            FROM populacao AS p
            LEFT JOIN casos_confirmados AS c
                ON p.{uf_col} = c.{uf_col} AND p.{ano_col} = c.{ano_col}
            ORDER BY p.{ano_col}, p.{uf_col}
        """
        return con.execute(query, [habitantes_base]).pl()
    finally:
        con.close()


def _codigo_municipio_normalizado(municipios_oficiais: pl.DataFrame) -> pl.DataFrame:
    """Monta a tabela de normalização de `ID_MUNICIP` para o código IBGE canônico.

    O layout do SINAN-DCA mudou por volta de 2007-2010 (mesma transição
    documentada em `features.COLUNAS_COM_ALIAS`): registros antigos trazem
    `ID_MUNICIP` com 6 dígitos (sem o dígito verificador do código IBGE),
    registros recentes trazem 7 dígitos (código IBGE completo). Sem
    normalizar, o mesmo município aparece como dois códigos "diferentes"
    conforme o ano -- foi essa duplicação, não códigos espúrios, que
    inflava a contagem de "municípios distintos" do datamart (ex.: PE
    chegava a 204 "municípios" para 185 municípios oficiais).

    Os 6 primeiros dígitos do código de 7 dígitos (o código de 7 dígitos
    menos o dígito verificador final) já identificam o município de forma
    única dentro da UF, então o prefixo de 6 dígitos é a chave de junção
    entre as duas eras.

    Args:
        municipios_oficiais: Saída de `load_municipios_oficiais` (colunas
            `codigo_ibge`, `uf`, `nome`).

    Returns:
        DataFrame com colunas `prefixo6` (6 dígitos, chave de junção com
        o `ID_MUNICIP` bruto, em qualquer uma das duas eras), `uf`
        (oficial, para validar contra a UF da partição) e `codigo_ibge`
        (7 dígitos, canônico).
    """
    return municipios_oficiais.with_columns(pl.col("codigo_ibge").str.slice(0, 6).alias("prefixo6"))


def aggregate_municipio_ano_lags(
    processed_dir: Path,
    lags: tuple[int, ...] = (1, 2),
    municipio_col: str = "ID_MUNICIP",
    uf_col: str = "uf",
    ano_col: str = "ano",
    municipios_oficiais: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Agrega casos confirmados por município-ano e cria lags temporais.

    A grade município-ano é construída a partir do universo OFICIAL de
    municípios de PE/PB/AL (`load_municipios_oficiais`, 510 municípios),
    não apenas dos códigos que aparecem em `processed_dir` -- caso
    contrário, municípios sem nenhuma notificação em todo o período ficam
    de fora do datamart, em vez de entrarem com `casos=0` (ver
    `_codigo_municipio_normalizado`).

    Cada `ID_MUNICIP` bruto (6 ou 7 dígitos, conforme a era do layout) é
    normalizado para o código IBGE canônico de 7 dígitos via junção pelo
    prefixo de 6 dígitos, e validado contra a UF oficial daquele código:
    registros cujo `ID_MUNICIP` não corresponde a nenhum município oficial,
    ou corresponde a um município de outra UF, são descartados do
    agregado (ver `relatorio_normalizacao_municipios` para quantificar).

    O restante da lógica é igual à versão anterior: conta os casos
    confirmados (CLASSI_FIN = "1") por município-ano em `processed_dir`,
    preenche com zero as combinações município/ano sem nenhum caso
    confirmado (para que os lags não "pulem" anos silenciosamente) e cria,
    via `LAG() OVER (PARTITION BY município ORDER BY ano)`, uma coluna por
    valor em `lags` com a contagem de casos N anos antes. A grade de anos
    é o intervalo contínuo entre o menor e o maior ano presentes em
    `processed_dir` (não apenas os anos com dados) -- há lacunas de
    notificação em alguns anos (ex.: 2007-2009 no Nordeste), e sem uma
    grade contínua o LAG(1) acabaria comparando anos não consecutivos sem
    avisar.

    Args:
        processed_dir: Diretório raiz dos dados em Parquet (data/processed/),
            já com o schema harmonizado (ver `prepare_partition_columns`).
        lags: Defasagens (em anos) a calcular. Por padrão, 1 e 2 anos.
        municipio_col: Coluna de município em `processed_dir`.
        uf_col: Coluna de UF em `processed_dir`.
        ano_col: Coluna de ano em `processed_dir`.
        municipios_oficiais: Lista oficial de municípios (ver
            `load_municipios_oficiais`); `None` carrega a lista padrão
            (PE/PB/AL) do arquivo versionado.

    Returns:
        DataFrame com uma linha por município-ano (município = código
        IBGE canônico de 7 dígitos): `municipio`, `uf`, `ano`, `casos` e
        uma coluna `casos_lag_<n>` por valor em `lags`.
    """
    mapa = _codigo_municipio_normalizado(municipios_oficiais or load_municipios_oficiais())
    colunas_lag = ",\n                ".join(
        f"LAG(casos, {n}) OVER (PARTITION BY municipio ORDER BY ano) AS casos_lag_{n}"
        for n in lags
    )
    con = duckdb.connect(":memory:")
    try:
        _read_processed_view(con, processed_dir)
        con.register("mapa_municipios", mapa)
        query = f"""
            WITH processed_normalizado AS (
                SELECT
                    p.*,
                    m.codigo_ibge AS municipio_normalizado
                FROM processed AS p
                LEFT JOIN mapa_municipios AS m
                    ON LEFT(p.{municipio_col}, 6) = m.prefixo6
                    AND p.{uf_col} = m.uf
            ),

            anos AS (
                SELECT UNNEST(RANGE(MIN({ano_col}), MAX({ano_col}) + 1)) AS ano
                FROM processed
            ),

            grade AS (
                SELECT
                    m.codigo_ibge AS municipio,
                    m.uf,
                    anos.ano
                FROM mapa_municipios AS m
                CROSS JOIN anos
            ),

            casos_confirmados AS (
                SELECT
                    municipio_normalizado AS municipio,
                    {ano_col} AS ano,
                    COUNT(*) AS casos
                FROM processed_normalizado
                WHERE CLASSI_FIN = '{CODIGO_CONFIRMADO}' AND municipio_normalizado IS NOT NULL
                GROUP BY municipio_normalizado, {ano_col}
            ),

            serie AS (
                SELECT
                    g.municipio,
                    g.uf,
                    g.ano,
                    COALESCE(c.casos, 0) AS casos
                FROM grade AS g
                LEFT JOIN casos_confirmados AS c
                    ON g.municipio = c.municipio AND g.ano = c.ano
            )

            SELECT
                municipio,
                uf,
                ano,
                casos,
                {colunas_lag}
            FROM serie
            ORDER BY municipio, ano
        """
        return con.execute(query).pl()
    finally:
        con.close()


def relatorio_normalizacao_municipios(
    processed_dir: Path,
    municipio_col: str = "ID_MUNICIP",
    uf_col: str = "uf",
    municipios_oficiais: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Quantifica o efeito da normalização/validação de `ID_MUNICIP`.

    Não é usada pelo pipeline em si (`aggregate_municipio_ano_lags` já
    aplica a normalização) -- serve só para auditoria/relatório: quantos
    códigos brutos distintos existem por UF, quantos municípios IBGE
    únicos eles de fato representam após normalizar, e quantos registros
    têm `ID_MUNICIP` que não corresponde a nenhum município oficial da
    mesma UF (descartados do agregado).

    Args:
        processed_dir: Diretório raiz dos dados em Parquet (data/processed/).
        municipio_col: Coluna de município em `processed_dir`.
        uf_col: Coluna de UF em `processed_dir`.
        municipios_oficiais: Ver `aggregate_municipio_ano_lags`.

    Returns:
        DataFrame com uma linha por UF: `uf`, `codigos_brutos_distintos`,
        `municipios_normalizados_distintos`, `registros_descartados`,
        `municipios_oficiais_sem_match`.
    """
    mapa = _codigo_municipio_normalizado(municipios_oficiais or load_municipios_oficiais())
    con = duckdb.connect(":memory:")
    try:
        _read_processed_view(con, processed_dir)
        con.register("mapa_municipios", mapa)
        query = f"""
            WITH juncao AS (
                SELECT
                    p.{uf_col} AS uf,
                    p.{municipio_col} AS codigo_bruto,
                    m.codigo_ibge AS municipio_normalizado
                FROM processed AS p
                LEFT JOIN mapa_municipios AS m
                    ON LEFT(p.{municipio_col}, 6) = m.prefixo6
                    AND p.{uf_col} = m.uf
            )
            SELECT
                uf,
                COUNT(DISTINCT codigo_bruto) AS codigos_brutos_distintos,
                COUNT(DISTINCT municipio_normalizado) AS municipios_normalizados_distintos,
                COUNT(*) FILTER (WHERE municipio_normalizado IS NULL) AS registros_descartados
            FROM juncao
            GROUP BY uf
            ORDER BY uf
        """
        resultado = con.execute(query).pl()
    finally:
        con.close()

    oficiais_por_uf = (municipios_oficiais or load_municipios_oficiais()).group_by("uf").len(
        name="municipios_oficiais"
    )
    presentes_por_uf = resultado.select("uf", "municipios_normalizados_distintos")
    sem_match = oficiais_por_uf.join(presentes_por_uf, on="uf", how="left").with_columns(
        (pl.col("municipios_oficiais") - pl.col("municipios_normalizados_distintos").fill_null(0)).alias(
            "municipios_oficiais_sem_notificacao"
        )
    ).select("uf", "municipios_oficiais_sem_notificacao")

    return resultado.join(sem_match, on="uf")


def label_municipio_ano_risk(
    df: pl.DataFrame,
    casos_col: str = "casos",
    limiar: int = LIMIAR_RISCO_PADRAO,
) -> pl.DataFrame:
    """Rotula a variável-alvo binária de risco por município-ano.

    Um município-ano é rotulado como "de risco" (1) quando teve ao menos
    `limiar` casos confirmados de Doença de Chagas aguda naquele ano; caso
    contrário, recebe rótulo 0. Por padrão (`limiar=1`), a presença de
    qualquer caso confirmado autóctone já caracteriza o município-ano como
    de risco -- esse agravo é raro o suficiente na região (ver
    `aggregate_municipio_ano_lags`) para que um limiar mais alto descarte
    quase todos os município-anos positivos.

    Args:
        df: DataFrame município-ano com a coluna de contagem de casos
            confirmados (tipicamente a saída de
            `aggregate_municipio_ano_lags`).
        casos_col: Coluna com a contagem de casos confirmados do ano.
        limiar: Número mínimo de casos confirmados para rotular o
            município-ano como de risco.

    Returns:
        DataFrame de entrada com a coluna `risco` adicionada (1 = de
        risco, 0 = sem risco).
    """
    return df.with_columns(
        (pl.col(casos_col) >= limiar).cast(pl.Int8).alias(COLUNA_RISCO)
    )


def build_risk_datamart(
    processed_dir: Path,
    populacao: pl.DataFrame | None = None,
    lags: tuple[int, ...] = (1, 2),
    limiar_risco: int = LIMIAR_RISCO_PADRAO,
) -> pl.DataFrame:
    """Monta o datamart analítico município-ano, já com o rótulo de risco.

    Agrega os casos confirmados por município-ano e seus lags temporais
    (`aggregate_municipio_ano_lags`, já sobre a grade OFICIAL dos 510
    municípios de PE/PB/AL), rotula a variável-alvo binária de risco
    (`label_municipio_ano_risk`) e, se `populacao` for informada, junta a
    população do MUNICÍPIO no ano (ver `fetch_ibge_population_municipal`)
    para derivar a taxa de incidência do ano por 100 mil habitantes.

    Antes, a população era agregada por UF (não por município): com só
    três UFs e 26 anos, UF combinada com ano funcionava quase como um
    identificador UF-ano, sem poder de discriminar municípios entre si.
    A tabela SIDRA 6579 usada aqui tem, sim, granularidade municipal (ver
    `fetch_ibge_population_municipal`).

    Args:
        processed_dir: Diretório raiz dos dados em Parquet (data/processed/),
            já com o schema harmonizado (ver `prepare_partition_columns`).
        populacao: População por município/ano (ver
            `fetch_ibge_population_municipal`; colunas `municipio`, `ano`,
            `populacao`); se None, a coluna `taxa_por_100k` não é criada.
        lags: Defasagens (em anos) a calcular, repassado a
            `aggregate_municipio_ano_lags`.
        limiar_risco: Repassado a `label_municipio_ano_risk`.

    Returns:
        DataFrame com uma linha por município-ano: `municipio`, `uf`,
        `ano`, `casos`, `casos_lag_<n>` por valor em `lags`, `risco` e,
        se `populacao` informada, `populacao` e `taxa_por_100k`.
    """
    datamart = aggregate_municipio_ano_lags(processed_dir, lags=lags)
    datamart = label_municipio_ano_risk(datamart, limiar=limiar_risco)

    if populacao is not None:
        pop = populacao.select("municipio", "ano", "populacao")
        datamart = datamart.join(pop, on=["municipio", "ano"], how="left")
        datamart = datamart.with_columns(
            (pl.col("casos") * 100_000.0 / pl.col("populacao")).alias("taxa_por_100k")
        )

    return datamart


def write_datamart_parquet(
    df: pl.DataFrame,
    output_dir: Path,
    uf_col: str = "uf",
    ano_col: str = "ano",
) -> None:
    """Persiste o datamart analítico em .parquet particionado por UF e ano.

    Ao contrário de `write_partitioned_parquet` (que grava as partições de
    notificação individual usando o escritor nativo do Polars), esta
    função grava o datamart município-ano/rótulo de risco explicitamente
    via PyArrow (`pyarrow.dataset.write_dataset`), no mesmo layout Hive
    `uf=<UF>/ano=<ANO>/` usado no restante do pipeline -- para que
    `query_processed`/`load_lazy_partitions` consigam ler esse datamart
    com o mesmo padrão de glob usado sobre data/processed/.

    `existing_data_behavior="delete_matching"` faz a escrita ser
    idempotente: reprocessar o mesmo UF/ano substitui a partição em vez
    de duplicar dados, igual a `write_partitioned_parquet`.

    Args:
        df: DataFrame do datamart a persistir (tipicamente a saída de
            `build_risk_datamart`). Precisa conter as colunas `uf_col` e
            `ano_col`.
        output_dir: Diretório raiz de saída do datamart (ex.:
            data/datamart/).
        uf_col: Nome da coluna usada para particionar por UF.
        ano_col: Nome da coluna usada para particionar por ano.
    """
    if df.is_empty():
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    tabela = df.to_arrow()
    particionamento = pa_dataset.partitioning(
        pa.schema([(uf_col, tabela.schema.field(uf_col).type), (ano_col, tabela.schema.field(ano_col).type)]),
        flavor="hive",
    )
    pa_dataset.write_dataset(
        tabela,
        base_dir=output_dir,
        format="parquet",
        partitioning=particionamento,
        basename_template=DATAMART_BASENAME_TEMPLATE,
        existing_data_behavior="delete_matching",
    )
