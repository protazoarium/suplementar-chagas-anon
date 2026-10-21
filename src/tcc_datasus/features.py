"""Engenharia de features em modo lazy com Polars."""

from pathlib import Path

import polars as pl

# O layout do SINAN-DCA mudou por volta de 2007-2010: alguns campos foram
# renomeados (ex.: CON_CLASSI -> CLASSI_FIN). As tuplas abaixo listam, em
# ordem de preferência, os nomes possíveis de cada campo conforme o ano.
COLUNA_CLASSIFICACAO_FINAL = ("CLASSI_FIN", "CON_CLASSI")
COLUNA_CRITERIO = ("CRITERIO", "CON_CRITER")
COLUNA_EVOLUCAO = ("EVOLUCAO", "CON_EVOLUC")
COLUNA_DUPLICIDADE = ("NDUPLIC_N", "NDUPLIC")
COLUNA_IDADE = ("NU_IDADE_N", "NU_IDADE")
COLUNA_DURACAO_SINTOMAS = ("TEMPO", "DURACAO")
COLUNA_ESCOLARIDADE = ("CS_ESCOL_N", "CS_ESCOLAR")

# Todas as colunas com nome variável entre eras, para uso por quem
# precisar renomear para o nome canônico (ex.: `transform.harmonize_schema`,
# necessário para consultas DuckDB uniformes sobre partições de anos
# diferentes).
COLUNAS_COM_ALIAS = (
    COLUNA_CLASSIFICACAO_FINAL,
    COLUNA_CRITERIO,
    COLUNA_EVOLUCAO,
    COLUNA_DUPLICIDADE,
    COLUNA_IDADE,
    COLUNA_ESCOLARIDADE,
)

# Códigos do SINAN para CLASSI_FIN/CON_CLASSI: 1=Confirmado, 2=Descartado
# (mantidos em ambas as versões do layout); os demais códigos (inconclusivo)
# e valores ausentes não servem de rótulo para um modelo supervisionado.
CODIGO_CONFIRMADO = "1"
CODIGO_DESCARTADO = "2"
CODIGOS_CLASSIFICACAO_UTIL = [CODIGO_CONFIRMADO, CODIGO_DESCARTADO]

# Código de notificação duplicada (Sim), em ambas as versões do layout.
CODIGO_DUPLICADO = "1"

# NU_IDADE_N/NU_IDADE combina uma unidade com um valor de 3 dígitos. A
# unidade é um dígito (1=hora, 2=dia, 3=mês, 4=ano) em arquivos a partir de
# ~2007; em arquivos mais antigos (layout DOS), é uma letra (H/D/M/A). "4"
# e "A" (ano) são equivalentes entre os dois formatos.
UNIDADE_IDADE_ANOS = "4"
MAPA_UNIDADE_IDADE_LEGADO = {"H": "1", "D": "2", "M": "3", "A": "4"}
IDADE_MAXIMA_PLAUSIVEL = 120
COLUNA_IDADE_ANOS = "IDADE_ANOS"

DEFAULT_COLUNAS_CATEGORICAS = (
    "CS_SEXO",
    "CS_RACA",
    COLUNA_ESCOLARIDADE,
    "CS_GESTANT",
    COLUNA_CRITERIO,
    COLUNA_EVOLUCAO,
)
DEFAULT_COLUNAS_NUMERICAS = (COLUNA_DURACAO_SINTOMAS,)
DEFAULT_GRUPO_CONDICIONANTE = ("SG_UF_NOT", "NU_ANO")

# CS_GESTANT não existe nos arquivos anteriores a ~2007 (não é uma
# renomeação -- o campo simplesmente não era coletado, ver
# `transform.load_lazy_partitions`); ao juntar eras, esses registros têm
# CS_GESTANT nulo. "9" (Ignorado) é o código do próprio domínio SINAN
# para essa situação, mais correto que uma imputação estatística aqui.
CODIGO_GESTANTE_IGNORADO = "9"

# CS_SEXO e a UF não são códigos numéricos do domínio SINAN (ao contrário
# de CS_RACA, CS_ESCOL_N, CS_GESTANT, que já são dígitos em texto), por
# isso precisam de um mapeamento explícito para virar feature numérica.
MAPA_SEXO = {"M": 1, "F": 2, "I": 3}
MAPA_UF = {"PE": 1, "PB": 2, "AL": 3}

COLUNA_LABEL = "label"


def _resolver_coluna(colunas_disponiveis: set[str], candidato: str | tuple[str, ...]) -> str | None:
    """Retorna o primeiro nome de coluna existente entre os candidatos.

    Os campos do SINAN-DCA mudaram de nome ao longo dos anos; esta função
    permite que as regras abaixo funcionem tanto no layout antigo quanto
    no atual, aplicando-se apenas às colunas de fato presentes no
    LazyFrame recebido (tipicamente um único ano/partição).
    """
    nomes = (candidato,) if isinstance(candidato, str) else candidato
    return next((nome for nome in nomes if nome in colunas_disponiveis), None)


def _idade_em_anos(coluna_idade: str) -> pl.Expr:
    """Extrai a idade em anos a partir do campo bruto de idade do SINAN.

    Aceita tanto o formato numérico (ex.: "4085" = 85 anos) quanto o
    formato legado com letra (ex.: "A027" = 27 anos). Retorna nulo quando
    a unidade não é "ano" (hora/dia/mês) ou o valor não pode ser
    interpretado (ex.: unidade "Ignorado").
    """
    texto = pl.col(coluna_idade).cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    unidade = texto.str.slice(0, 1).replace_strict(
        MAPA_UNIDADE_IDADE_LEGADO, default=texto.str.slice(0, 1)
    )
    valor = texto.str.slice(1).cast(pl.Int64, strict=False)
    return pl.when(unidade == UNIDADE_IDADE_ANOS).then(valor).otherwise(None)


def apply_epidemiological_filters(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Aplica regras epidemiológicas de consistência e elegibilidade.

    Regras aplicadas (cada uma só é aplicada se a coluna correspondente
    existir no LazyFrame recebido):

    - Mantém apenas registros com classificação final Confirmado ou
      Descartado (CLASSI_FIN/CON_CLASSI em {"1", "2"}); descarta casos
      inconclusivos ou sem classificação, pois não oferecem um rótulo
      utilizável para o modelo preditivo.
    - Descarta notificações duplicadas (NDUPLIC_N/NDUPLIC == "1").
    - Descarta inconsistência temporal: data de início dos sintomas
      posterior à data de notificação (DT_SIN_PRI > DT_NOTIFIC).
    - Descarta idade implausível: mais de 120 anos (ver `_idade_em_anos`).

    Registros com valores ausentes nos campos usados por cada regra não
    são descartados por causa disso (a ausência é tratada na imputação),
    exceto para a classificação final, cuja ausência já significa que o
    registro não tem rótulo.

    Args:
        lf: LazyFrame de entrada com os dados brutos/limpos de um
            ano/partição do SINAN-DCA.

    Returns:
        LazyFrame filtrado.
    """
    colunas = set(lf.collect_schema().names())

    coluna_classi = _resolver_coluna(colunas, COLUNA_CLASSIFICACAO_FINAL)
    if coluna_classi is not None:
        lf = lf.filter(pl.col(coluna_classi).is_in(CODIGOS_CLASSIFICACAO_UTIL))

    coluna_duplic = _resolver_coluna(colunas, COLUNA_DUPLICIDADE)
    if coluna_duplic is not None:
        lf = lf.filter(
            pl.col(coluna_duplic).is_null() | (pl.col(coluna_duplic) != CODIGO_DUPLICADO)
        )

    if {"DT_SIN_PRI", "DT_NOTIFIC"} <= colunas:
        lf = lf.filter(
            pl.col("DT_SIN_PRI").is_null()
            | pl.col("DT_NOTIFIC").is_null()
            | (pl.col("DT_SIN_PRI") <= pl.col("DT_NOTIFIC"))
        )

    coluna_idade = _resolver_coluna(colunas, COLUNA_IDADE)
    if coluna_idade is not None:
        idade_anos = _idade_em_anos(coluna_idade)
        # Idade fora do range de anos (hora/dia/mês) ou não interpretável
        # não é considerada implausível aqui -- só quando expressa em anos
        # e acima do limite. A ausência de idade também não descarta o
        # registro (tratada na imputação).
        lf = lf.filter(idade_anos.is_null() | (idade_anos <= IDADE_MAXIMA_PLAUSIVEL))

    return lf


def impute_missing_values(
    lf: pl.LazyFrame,
    colunas_categoricas: tuple[str | tuple[str, ...], ...] = DEFAULT_COLUNAS_CATEGORICAS,
    colunas_numericas: tuple[str | tuple[str, ...], ...] = DEFAULT_COLUNAS_NUMERICAS,
    grupo: tuple[str, ...] = DEFAULT_GRUPO_CONDICIONANTE,
    incluir_idade: bool = True,
) -> pl.LazyFrame:
    """Imputa valores ausentes por moda (categóricas) ou mediana (numéricas).

    A imputação é condicionada por grupo (por padrão, UF de notificação e
    ano): calcula-se a moda/mediana dentro de cada grupo e usa-se esse
    valor para preencher os nulos do grupo. Quando um grupo não tem nenhum
    valor não nulo para preencher (ex.: partição muito pequena), usa-se a
    moda/mediana global da coluna como reserva.

    Args:
        lf: LazyFrame de entrada (tipicamente já filtrado por
            `apply_epidemiological_filters`).
        colunas_categoricas: Colunas (ou tuplas de nomes alternativos, para
            lidar com a mudança de layout do SINAN ao longo dos anos) a
            imputar por moda condicionada.
        colunas_numericas: Colunas (ou tuplas de nomes alternativos) a
            imputar por mediana condicionada.
        grupo: Colunas usadas para condicionar a moda/mediana. Colunas
            ausentes no LazyFrame são ignoradas.
        incluir_idade: Se True (padrão) e o campo de idade bruto
            (NU_IDADE_N/NU_IDADE) existir, deriva a coluna `IDADE_ANOS`
            (idade em anos, ver `_idade_em_anos`) e a imputa por mediana
            condicionada -- o campo bruto combina unidade e valor, então
            não faz sentido tirar a mediana dele diretamente.

    Returns:
        LazyFrame com as colunas informadas imputadas (e, quando
        aplicável, a coluna `IDADE_ANOS` adicionada).
    """
    colunas = set(lf.collect_schema().names())
    grupo_presente = [g for g in grupo if g in colunas]

    def _sobre_grupo(expr: pl.Expr) -> pl.Expr:
        return expr.over(grupo_presente) if grupo_presente else expr

    if incluir_idade:
        coluna_idade = _resolver_coluna(colunas, COLUNA_IDADE)
        if coluna_idade is not None:
            lf = lf.with_columns(_idade_em_anos(coluna_idade).alias(COLUNA_IDADE_ANOS))
            colunas = colunas | {COLUNA_IDADE_ANOS}
            colunas_numericas = (*colunas_numericas, COLUNA_IDADE_ANOS)

    exprs = []

    for candidato in colunas_numericas:
        nome = _resolver_coluna(colunas, candidato)
        if nome is None:
            continue
        # Campos numéricos podem estar armazenados como texto em anos mais
        # antigos do SINAN (ver `dbf_to_dataframe`/`convert.py`).
        coluna = pl.col(nome).cast(pl.Int64, strict=False)
        base = coluna.drop_nulls()
        mediana_condicionada = _sobre_grupo(base.median())
        mediana_global = base.median()
        exprs.append(coluna.fill_null(mediana_condicionada).fill_null(mediana_global).alias(nome))

    for candidato in colunas_categoricas:
        nome = _resolver_coluna(colunas, candidato)
        if nome is None:
            continue
        base = pl.col(nome).drop_nulls()
        moda_condicionada = _sobre_grupo(base.mode().first())
        moda_global = base.mode().first()
        exprs.append(pl.col(nome).fill_null(moda_condicionada).fill_null(moda_global).alias(nome))

    return lf.with_columns(exprs) if exprs else lf


def load_lazy_partitions(
    processed_dir: Path,
    ufs: list[str] | None = None,
    anos: list[int] | None = None,
) -> pl.LazyFrame:
    """Carrega partições selecionadas de data/processed/ como LazyFrame.

    As colunas de partição (uf, ano) são inferidas do próprio caminho
    (layout uf=<UF>/ano=<ANO>/, ver `transform.write_partitioned_parquet`).

    Mesmo após a harmonização de schema entre eras do SINAN-DCA (ver
    `transform.harmonize_schema`), alguns campos existem só numa das eras
    (ex.: CS_GESTANT não existe nos arquivos anteriores a ~2007 -- não é
    um caso de renomeação, o campo simplesmente não era coletado). Passar
    `missing_columns="insert"` sozinho não resolve isso: o Polars usa o
    schema de um dos arquivos como referência e descarta silenciosamente
    colunas que não estejam nele, em vez de unir os schemas de todos os
    arquivos. Por isso o schema de leitura aqui é construído explicitamente
    como a união dos schemas de todos os arquivos (uma operação barata,
    só lê metadados) antes de montar o LazyFrame.

    Args:
        processed_dir: Diretório raiz dos dados em Parquet (data/processed/).
        ufs: UFs a incluir; None inclui todas as disponíveis.
        anos: Anos a incluir; None inclui todos os disponíveis.

    Returns:
        LazyFrame do Polars referenciando as partições selecionadas, sem
        materializar os dados em memória. Colunas ausentes num arquivo
        (mas presentes em outro) aparecem como nulas nesse arquivo.
    """
    arquivos = sorted(processed_dir.glob("**/*.parquet"))
    schema_uniao: dict[str, pl.DataType] = {}
    for arquivo in arquivos:
        for nome, tipo in pl.read_parquet_schema(arquivo).items():
            schema_uniao.setdefault(nome, tipo)

    lf = pl.scan_parquet(
        processed_dir / "**" / "*.parquet",
        hive_partitioning=True,
        schema=schema_uniao or None,
        missing_columns="insert",
    )
    if ufs is not None:
        lf = lf.filter(pl.col("uf").is_in(ufs))
    if anos is not None:
        lf = lf.filter(pl.col("ano").is_in(anos))
    return lf


def build_features(
    lf: pl.LazyFrame,
    processed_dir: Path,
    populacao: pl.DataFrame | None = None,
) -> pl.LazyFrame:
    """Constrói as features usadas no treino dos modelos.

    Parte de um LazyFrame já filtrado/imputado ao nível de notificação
    (tipicamente a saída de `load_lazy_partitions` sobre data/processed/)
    e junta o que foi construído nas demais etapas do projeto:

    - Sazonalidade: mês da notificação (DT_NOTIFIC).
    - Casos confirmados no município nos 1 e 2 anos anteriores
      (`transform.aggregate_municipio_ano_lags`, via SQL/DuckDB) --
      sinal de risco epidemiológico local, sem vazamento (usa só anos
      passados).
    - Taxa de incidência defasada por 100 mil habitantes: casos
      confirmados do município 1/2 anos atrás sobre a população da UF no
      ano corrente (só se `populacao` for informado; ver
      `transform.fetch_ibge_population`).

    Os campos categóricos são codificados como inteiros: a maioria dos
    campos do SINAN já é um código numérico do próprio domínio (ex.:
    CS_RACA "1".."5"), então basta converter de texto para inteiro; só
    CS_SEXO e a UF, que não são numéricos, usam um mapeamento fixo
    (`MAPA_SEXO`/`MAPA_UF`) -- um mapeamento fixo evita a armadilha de
    usar códigos categóricos "físicos" do Polars, que não são garantidos
    estáveis entre execuções sem um cache de strings global.

    Deliberadamente NÃO usa CRITERIO, EVOLUCAO nem uma taxa por 100 mil
    do próprio ano corrente como features: são preenchidos como parte do
    processo de classificação/desfecho do caso (ou, no caso da taxa do
    ano corrente, derivados de uma contagem que inclui o próprio caso),
    o que vazaria informação do rótulo para o modelo.

    Args:
        lf: LazyFrame de entrada, já filtrado/imputado (ver
            `apply_epidemiological_filters`, `impute_missing_values`,
            `load_lazy_partitions`).
        processed_dir: Diretório raiz dos dados em Parquet (data/processed/),
            usado para calcular os lags município-ano via DuckDB. Deve
            corresponder aos mesmos dados carregados em `lf`.
        populacao: População por UF/ano (ver
            `transform.fetch_ibge_population`); se None, as features de
            taxa por 100 mil não são criadas.

    Returns:
        LazyFrame com uma linha por notificação: as colunas de features
        (numéricas) e a coluna de rótulo `label` (1 = Confirmado,
        0 = Descartado).
    """
    # Import tardio: `transform` importa `features` (para harmonizar o
    # schema e filtrar casos confirmados), então importar no nível do
    # módulo aqui criaria um ciclo.
    from tcc_datasus import transform

    lags = (
        transform.aggregate_municipio_ano_lags(processed_dir)
        .lazy()
        .select(
            pl.col("municipio").alias("ID_MUNICIP"),
            pl.col("ano"),
            pl.col("casos_lag_1"),
            pl.col("casos_lag_2"),
        )
    )
    lf = lf.join(lags, on=["ID_MUNICIP", "ano"], how="left")
    lf = lf.with_columns(
        pl.col("casos_lag_1").fill_null(0),
        pl.col("casos_lag_2").fill_null(0),
    )

    if populacao is not None:
        pop_lf = populacao.lazy().select("uf", "ano", "populacao")
        lf = lf.join(pop_lf, on=["uf", "ano"], how="left")
        lf = lf.with_columns(
            (pl.col("casos_lag_1") * 100_000.0 / pl.col("populacao")).alias("TAXA_LAG_1_POR_100K"),
            (pl.col("casos_lag_2") * 100_000.0 / pl.col("populacao")).alias("TAXA_LAG_2_POR_100K"),
        )

    lf = lf.with_columns(
        (pl.col("CLASSI_FIN") == CODIGO_CONFIRMADO).cast(pl.Int64).alias(COLUNA_LABEL),
        pl.col("DT_NOTIFIC").dt.month().alias("MES_NOTIFIC"),
        pl.col("CS_SEXO").replace_strict(MAPA_SEXO, default=None, return_dtype=pl.Int64).alias("CS_SEXO_COD"),
        pl.col("CS_RACA").cast(pl.Int64, strict=False).alias("CS_RACA_COD"),
        pl.col(COLUNA_ESCOLARIDADE[0]).cast(pl.Int64, strict=False).alias("CS_ESCOL_COD"),
        pl.col("CS_GESTANT")
        .fill_null(CODIGO_GESTANTE_IGNORADO)
        .cast(pl.Int64, strict=False)
        .alias("CS_GESTANT_COD"),
        pl.col("uf").replace_strict(MAPA_UF, default=None, return_dtype=pl.Int64).alias("UF_COD"),
    )

    colunas_finais = [
        "MES_NOTIFIC",
        COLUNA_IDADE_ANOS,
        "CS_SEXO_COD",
        "CS_RACA_COD",
        "CS_ESCOL_COD",
        "CS_GESTANT_COD",
        "UF_COD",
        "casos_lag_1",
        "casos_lag_2",
    ]
    if populacao is not None:
        colunas_finais += ["populacao", "TAXA_LAG_1_POR_100K", "TAXA_LAG_2_POR_100K"]
    colunas_finais.append(COLUNA_LABEL)

    return lf.select(colunas_finais)
