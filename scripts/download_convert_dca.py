"""Baixa os dados do SINAN-DCA (Doença de Chagas Aguda) do FTP do DATASUS
e converte os .dbc para .csv, restritos a PE, PB e AL.

O SINAN-DCA é distribuído pelo DATASUS apenas em arquivos nacionais
(CHAGBR<AA>.dbc, sem particionamento por UF); o recorte para PE/PB/AL é
aplicado durante a conversão, a partir da coluna SG_UF_NOT.

Uso:
    uv run python scripts/download_convert_dca.py
    uv run python scripts/download_convert_dca.py --anos 2015 2016 2017
    uv run python scripts/download_convert_dca.py --fonte PRELIM
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tcc_datasus.convert import dbc_to_dbf, dbf_to_dataframe
from tcc_datasus.download import download_all, list_available_years

UFS = ["PE", "PB", "AL"]
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
CSV_DIR = INTERIM_DIR / "csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anos",
        type=int,
        nargs="+",
        default=None,
        help="Anos a baixar; padrão: todos os disponíveis na fonte escolhida.",
    )
    parser.add_argument(
        "--fonte",
        choices=["FINAIS", "PRELIM"],
        default="FINAIS",
        help="Dados consolidados (FINAIS) ou preliminares/sujeitos a revisão (PRELIM).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Baixa novamente mesmo que o .dbc já exista em data/raw/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anos = args.anos or list_available_years(args.fonte)
    print(f"SINAN-DCA ({args.fonte}): baixando anos {anos}")

    dbc_paths = download_all(anos, RAW_DIR, source=args.fonte, overwrite=args.overwrite)

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    for dbc_path in dbc_paths:
        dbf_path = dbc_to_dbf(dbc_path, INTERIM_DIR)
        df = dbf_to_dataframe(dbf_path, ufs=UFS)
        csv_path = CSV_DIR / f"{dbc_path.stem}_PE_PB_AL.csv"
        df.write_csv(csv_path)
        print(f"  {dbc_path.name}: {df.height} registros (PE/PB/AL) -> {csv_path}")


if __name__ == "__main__":
    main()
