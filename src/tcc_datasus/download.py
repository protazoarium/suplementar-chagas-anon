"""Download dos arquivos brutos do SINAN/DATASUS (.dbc)."""

from __future__ import annotations

import ftplib
import re
from pathlib import Path

FTP_HOST = "ftp.datasus.gov.br"

# O agravo "Doença de Chagas Aguda" (SINAN-DCA) é disponibilizado pelo
# DATASUS apenas em arquivos nacionais (sufixo "BR"), sem particionamento
# por UF -- diferente de outros agravos do SINAN (ex.: dengue). O recorte
# para PE, PB e AL é feito depois, na conversão (ver convert.py), a partir
# da coluna SG_UF_NOT presente nos próprios registros.
AGRAVO_PREFIX = "CHAGBR"

# "FINAIS" contém os dados consolidados; "PRELIM" contém os anos mais
# recentes, ainda sujeitos a revisão pelo Ministério da Saúde.
FTP_DIRS = {
    "FINAIS": "/dissemin/publicos/SINAN/DADOS/FINAIS",
    "PRELIM": "/dissemin/publicos/SINAN/DADOS/PRELIM",
}


def _remote_filename(ano: int) -> str:
    return f"{AGRAVO_PREFIX}{ano % 100:02d}.dbc"


def _retrieve(ftp: ftplib.FTP, remote_name: str, dest_path: Path) -> None:
    """Baixa um arquivo via FTP para um arquivo temporário e o promove
    atomicamente, evitando deixar um .dbc corrompido caso a conexão caia."""
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        with open(tmp_path, "wb") as f:
            ftp.retrbinary(f"RETR {remote_name}", f.write)
        tmp_path.replace(dest_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def list_available_years(source: str = "FINAIS") -> list[int]:
    """Lista os anos com arquivo do SINAN-DCA disponível no FTP do DATASUS.

    Args:
        source: "FINAIS" para dados consolidados ou "PRELIM" para dados
            preliminares (anos mais recentes, ainda sujeitos a revisão).

    Returns:
        Anos (4 dígitos) disponíveis para download, em ordem crescente.
    """
    pattern = re.compile(rf"^{AGRAVO_PREFIX}(\d{{2}})\.dbc$", re.IGNORECASE)
    with ftplib.FTP(FTP_HOST, timeout=30) as ftp:
        ftp.login()
        ftp.cwd(FTP_DIRS[source])
        names = ftp.nlst()

    anos = {2000 + int(m.group(1)) for name in names if (m := pattern.match(name))}
    return sorted(anos)


def download_dbc(
    ano: int,
    dest_dir: Path,
    source: str = "FINAIS",
    overwrite: bool = False,
) -> Path:
    """Baixa o arquivo .dbc nacional do SINAN-DCA para um ano específico.

    O SINAN-DCA não é distribuído por UF: o arquivo baixado cobre o Brasil
    inteiro. O recorte para PE, PB e AL acontece na etapa de conversão.

    Args:
        ano: Ano de referência dos dados.
        dest_dir: Diretório de destino do arquivo baixado (data/raw/).
        source: "FINAIS" ou "PRELIM" (ver `FTP_DIRS`).
        overwrite: Se True, baixa novamente mesmo que o arquivo já exista.

    Returns:
        Caminho local do arquivo .dbc baixado.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    remote_name = _remote_filename(ano)
    dest_path = dest_dir / remote_name

    if dest_path.exists() and not overwrite:
        return dest_path

    with ftplib.FTP(FTP_HOST, timeout=30) as ftp:
        ftp.login()
        ftp.cwd(FTP_DIRS[source])
        _retrieve(ftp, remote_name, dest_path)

    return dest_path


def download_all(
    anos: list[int],
    dest_dir: Path,
    source: str = "FINAIS",
    overwrite: bool = False,
) -> list[Path]:
    """Baixa os arquivos .dbc nacionais do SINAN-DCA para os anos informados.

    Reutiliza uma única conexão FTP para todos os anos.

    Args:
        anos: Anos de referência a baixar.
        dest_dir: Diretório de destino dos arquivos baixados (data/raw/).
        source: "FINAIS" ou "PRELIM" (ver `FTP_DIRS`).
        overwrite: Se True, baixa novamente mesmo que o arquivo já exista.

    Returns:
        Caminhos locais de todos os arquivos .dbc baixados.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    with ftplib.FTP(FTP_HOST, timeout=30) as ftp:
        ftp.login()
        ftp.cwd(FTP_DIRS[source])
        for ano in anos:
            remote_name = _remote_filename(ano)
            dest_path = dest_dir / remote_name
            if not dest_path.exists() or overwrite:
                _retrieve(ftp, remote_name, dest_path)
            paths.append(dest_path)

    return paths
