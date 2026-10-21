"""Testes de integridade do datamart analítico município-ano (rótulo de risco)."""

from pathlib import Path

import polars as pl
import pyarrow.dataset as pa_dataset
import pytest

from tcc_datasus.transform import (
    COLUNA_RISCO,
    build_risk_datamart,
    label_municipio_ano_risk,
    write_datamart_parquet,
    write_partitioned_parquet,
)

# Dois municípios de PE: 260001 tem 1 caso confirmado em 2018 e nenhum em
# 2019 (1 descartado); 260002 tem 1 caso confirmado em cada ano.
NOTIFICACOES = pl.DataFrame(
    {
        "ID_MUNICIP": ["260001", "260001", "260002", "260002"],
        "CLASSI_FIN": ["1", "2", "1", "1"],
        "uf": ["PE", "PE", "PE", "PE"],
        "ano": [2018, 2019, 2018, 2019],
    }
)


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    destino = tmp_path / "processed"
    write_partitioned_parquet(NOTIFICACOES, destino)
    return destino


def test_label_municipio_ano_risk_marca_de_risco_quando_ha_caso_confirmado():
    df = pl.DataFrame({"casos": [0, 1, 5]})
    rotulado = label_municipio_ano_risk(df)
    assert rotulado[COLUNA_RISCO].to_list() == [0, 1, 1]


def test_label_municipio_ano_risk_respeita_limiar_customizado():
    df = pl.DataFrame({"casos": [0, 1, 2, 3]})
    rotulado = label_municipio_ano_risk(df, limiar=3)
    assert rotulado[COLUNA_RISCO].to_list() == [0, 0, 0, 1]


def test_label_municipio_ano_risk_e_binaria():
    df = pl.DataFrame({"casos": list(range(10))})
    rotulado = label_municipio_ano_risk(df)
    assert set(rotulado[COLUNA_RISCO].unique().to_list()) <= {0, 1}


def test_build_risk_datamart_uma_linha_por_municipio_ano(processed_dir: Path):
    datamart = build_risk_datamart(processed_dir)
    assert datamart.select(["municipio", "ano"]).is_duplicated().sum() == 0


def test_build_risk_datamart_rotulo_consistente_com_casos(processed_dir: Path):
    datamart = build_risk_datamart(processed_dir)
    esperado = datamart.select((pl.col("casos") >= 1).cast(pl.Int8))
    assert datamart[COLUNA_RISCO].to_list() == esperado.to_series().to_list()


def test_build_risk_datamart_reflete_casos_conhecidos(processed_dir: Path):
    datamart = build_risk_datamart(processed_dir)
    linha = datamart.filter((pl.col("municipio") == "260001") & (pl.col("ano") == 2019))
    assert linha["casos"].item() == 0
    assert linha[COLUNA_RISCO].item() == 0
    linha = datamart.filter((pl.col("municipio") == "260002") & (pl.col("ano") == 2019))
    assert linha["casos"].item() == 1
    assert linha[COLUNA_RISCO].item() == 1


def test_build_risk_datamart_com_populacao_adiciona_taxa(processed_dir: Path):
    populacao = pl.DataFrame({"uf": ["PE", "PE"], "ano": [2018, 2019], "populacao": [9_000_000, 9_100_000]})
    datamart = build_risk_datamart(processed_dir, populacao=populacao)
    assert "taxa_por_100k" in datamart.columns
    assert (datamart["taxa_por_100k"] >= 0).all()


def test_write_datamart_parquet_gera_layout_hive_uf_ano(tmp_path: Path, processed_dir: Path):
    datamart = build_risk_datamart(processed_dir)
    destino = tmp_path / "datamart"
    write_datamart_parquet(datamart, destino)

    particoes = sorted(p.relative_to(destino).parent.as_posix() for p in destino.rglob("*.parquet"))
    assert particoes == ["uf=PE/ano=2018", "uf=PE/ano=2019"]


def test_write_datamart_parquet_roundtrip_preserva_linhas_e_particoes(tmp_path: Path, processed_dir: Path):
    datamart = build_risk_datamart(processed_dir)
    destino = tmp_path / "datamart"
    write_datamart_parquet(datamart, destino)

    lido = pa_dataset.dataset(destino, format="parquet", partitioning="hive").to_table().to_pandas()

    assert len(lido) == len(datamart)
    assert set(lido[COLUNA_RISCO].unique().tolist()) <= {0, 1}
    # As colunas de partição do Hive (derivadas do caminho) precisam bater
    # com os valores de uf/ano que a própria linha carrega.
    for uf, ano in zip(lido["uf"], lido["ano"], strict=True):
        assert uf == "PE"
        assert ano in (2018, 2019)


def test_write_datamart_parquet_e_idempotente(tmp_path: Path, processed_dir: Path):
    datamart = build_risk_datamart(processed_dir)
    destino = tmp_path / "datamart"
    write_datamart_parquet(datamart, destino)
    write_datamart_parquet(datamart, destino)

    lido = pa_dataset.dataset(destino, format="parquet", partitioning="hive").to_table()
    assert lido.num_rows == len(datamart)


def test_write_datamart_parquet_dataframe_vazio_nao_cria_diretorio(tmp_path: Path):
    vazio = pl.DataFrame(
        {"municipio": [], "uf": [], "ano": [], "casos": [], COLUNA_RISCO: []},
        schema={"municipio": pl.Utf8, "uf": pl.Utf8, "ano": pl.Int64, "casos": pl.Int64, COLUNA_RISCO: pl.Int8},
    )
    destino = tmp_path / "datamart_vazio"
    write_datamart_parquet(vazio, destino)
    assert not destino.exists()
