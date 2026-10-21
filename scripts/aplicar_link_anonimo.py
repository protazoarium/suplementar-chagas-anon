"""Troca o marcador [LINK-ANONIMIZADO-PENDENTE] pelo link real em docx/pptx/md e refaz os PDFs.

Uso: python3 scripts/aplicar_link_anonimo.py https://anonymous.4open.science/r/<id>
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

MARCADOR = "[LINK-ANONIMIZADO-PENDENTE]"
ALVOS = ["artigo_final_revisado.docx", "carta_de_ajustes_e_explicacao_das_mudancas.docx",
         "apresentacao/apresentacao_SBTI2026.pptx", "apresentacao/roteiro_apresentacao.docx", "README.md"]
link = sys.argv[1]
for a in map(Path, ALVOS):
    if not a.exists():
        continue
    if a.suffix == ".md":
        a.write_text(a.read_text(encoding="utf8").replace(MARCADOR, link), encoding="utf8")
        continue
    tmp = a.with_suffix(a.suffix + ".tmp")
    with zipfile.ZipFile(a) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for it in zin.infolist():
            dados = zin.read(it.filename)
            if it.filename.endswith(".xml"):
                dados = dados.decode("utf8").replace(MARCADOR, link).encode("utf8")
            zout.writestr(it, dados)
    shutil.move(tmp, a)
    if a.suffix == ".pptx":
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(a.parent), str(a)], check=True)
