"""Make local copies of the Colab notebooks (the originals are never edited).

The notebooks on Drive mount Google Drive and look for the project under
/content/drive/MyDrive/.../blood-cancer-early-detection.  For every
notebooks/NN_name.ipynb this script writes notebooks/NN_name.local.ipynb in which:

* ``from google.colab import drive`` imports a no-op stand-in
  (notebooks/local_colab_shim.py) instead of Google's module;
* every "/content/drive/MyDrive" path points at the folder that contains
  leukemia-prediagnose/ on this computer;
* ``import pandas as pd`` is added to the first code cell when the notebook
  uses ``pd.`` but never imports it (notebook 13 on Drive relies on an
  earlier Colab session for this).

Nothing else changes.  Run it again whenever you move the project folder.

    python scripts/make_local_notebooks.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../blood-cancer-early-detection
MYDRIVE = ROOT.parent.parent                        # folder that holds leukemia-prediagnose/
NB_DIR = ROOT / "notebooks"

SHIM = '''"""Local stand-in for google.colab.drive, used by the *.local.ipynb notebooks."""


class _Drive:
    @staticmethod
    def mount(*args, **kwargs):
        print("[local] Google Drive mount skipped; using the local project folder")


drive = _Drive()
'''

COLAB_IMPORT = re.compile(r"from\s+google\.colab\s+import\s+drive")
PANDAS_IMPORT = re.compile(r"^\s*import\s+pandas\s+as\s+pd\b|^\s*import\s+.*\bpandas as pd\b", re.M)


def cell_text(cell) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def localize(text: str) -> str:
    text = COLAB_IMPORT.sub("from local_colab_shim import drive", text)
    return text.replace("/content/drive/MyDrive", MYDRIVE.as_posix())


def main() -> None:
    if ROOT.parent.name != "leukemia-prediagnose":
        print("WARNING: expected the project at .../leukemia-prediagnose/blood-cancer-early-detection; "
              f"found parent folder {ROOT.parent.name!r}. Notebooks that search MyDrive may not find it.")
    (NB_DIR / "local_colab_shim.py").write_text(SHIM, encoding="utf-8")
    made = []
    for nb_path in sorted(NB_DIR.glob("*.ipynb")):
        if nb_path.name.endswith(".local.ipynb") or nb_path.name.endswith(".executed.ipynb"):
            continue
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        for cell in code:
            text = cell_text(cell)
            new = localize(text)
            if new != text:
                cell["source"] = new.splitlines(keepends=True)
        all_code = "\n".join(cell_text(c) for c in code)
        if code and re.search(r"\bpd\.", all_code) and not PANDAS_IMPORT.search(all_code):
            code[0]["source"] = ("import pandas as pd  # added by make_local_notebooks.py\n"
                                 + cell_text(code[0])).splitlines(keepends=True)
        for cell in code:                      # never carry stale Colab outputs
            cell["outputs"], cell["execution_count"] = [], None
        out = nb_path.with_name(nb_path.stem + ".local.ipynb")
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        made.append(out.name)
    print(f"wrote {len(made)} local notebooks next to the originals in {NB_DIR}")
    print(f"MyDrive stand-in: {MYDRIVE}")


if __name__ == "__main__":
    main()
