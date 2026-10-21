"""Conversão dos arquivos brutos .dbc/.dbf para DataFrames."""

from pathlib import Path

import polars as pl
import pyreaddbc
from dbfread import DBF

# Códigos de UF do IBGE usados no campo SG_UF_NOT dos registros do SINAN.
UF_IBGE_CODE = {"PE": "26", "PB": "25", "AL": "27"}

DBF_ENCODING = "iso-8859-1"
DBF_DATE_FORMAT = "%Y%m%d"


def dbc_to_dbf(dbc_path: Path, dest_dir: Path) -> Path:
    """Descompacta um arquivo .dbc do DATASUS para o formato .dbf.

    Args:
        dbc_path: Caminho do arquivo .dbc de origem (data/raw/).
        dest_dir: Diretório de destino do .dbf gerado (data/interim/).

    Returns:
        Caminho do arquivo .dbf gerado.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dbf_path = dest_dir / f"{dbc_path.stem}.dbf"
    pyreaddbc.dbc2dbf(str(dbc_path), str(dbf_path))
    return dbf_path


def _decode(valor: bytes | None) -> str | None:
    if valor is None:
        return None
    texto = valor.decode(DBF_ENCODING, errors="replace").strip()
    return texto or None


def dbf_to_dataframe(dbf_path: Path, ufs: list[str] | None = None) -> pl.DataFrame:
    """Lê um arquivo .dbf e o converte para um DataFrame Polars.

    O dbfread é usado só para decodificar os bytes brutos de cada campo
    (`raw=True`); todo o parsing de datas e a conversão de tipos numéricos
    são feitos explicitamente com expressões Polars a partir do schema do
    próprio .dbf (campos tipo "D" viram Date, tipo "N" viram Int64), em vez
    de depender da inferência automática do dbfread ou do Polars.

    Os registros são filtrados durante a leitura (streaming, via dbfread),
    sem materializar o arquivo inteiro em memória antes do filtro.

    Args:
        dbf_path: Caminho do arquivo .dbf a ser lido.
        ufs: Siglas de UF para restringir os registros pela coluna
            SG_UF_NOT (ex.: ["PE", "PB", "AL"]). None mantém todo o Brasil.

    Returns:
        DataFrame Polars com os registros do arquivo (filtrados por UF,
        se informado), com colunas de data e numéricas já tipadas.
    """
    # Conforme o ano, SG_UF_NOT vem como código IBGE ("26") ou como sigla
    # ("PE") -- o layout do SINAN mudou ao longo do tempo. Aceita os dois.
    codigos = None
    if ufs:
        codigos = {UF_IBGE_CODE[uf] for uf in ufs} | set(ufs)

    tabela = DBF(str(dbf_path), encoding=DBF_ENCODING, load=False, raw=True)

    linhas = []
    for registro in tabela:
        linha = {nome: _decode(valor) for nome, valor in registro.items()}
        if codigos is not None and linha.get("SG_UF_NOT") not in codigos:
            continue
        linhas.append(linha)

    # Schema explícito (tudo Utf8) em vez de inferido: evita qualquer
    # ambiguidade de tipo entre arquivos de anos com layouts diferentes,
    # e deixa o parsing de fato a cargo dos casts abaixo.
    schema = {campo.name: pl.Utf8 for campo in tabela.fields}
    df = pl.DataFrame(linhas, schema=schema)

    colunas_data = [f.name for f in tabela.fields if f.type == "D" and f.name in df.columns]
    colunas_numericas = [f.name for f in tabela.fields if f.type == "N" and f.name in df.columns]

    return df.with_columns(
        [pl.col(c).str.strptime(pl.Date, DBF_DATE_FORMAT, strict=False) for c in colunas_data]
        + [pl.col(c).cast(pl.Int64, strict=False) for c in colunas_numericas]
    )
