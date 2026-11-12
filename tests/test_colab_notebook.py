"""The Colab notebook must embed the CURRENT tryon_server code and contain only valid Python.

Fails if someone edits tryon_server/ and forgets to run scripts/build_colab_notebook.py.
"""

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "colab_tryon_server.ipynb"


def code_cells():
    return [c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells if c.cell_type == "code"]


def test_embedded_server_files_are_up_to_date():
    embedded = {}
    for source in code_cells():
        if source.startswith("%%writefile"):
            header, body = source.split("\n", 1)
            embedded[header.rsplit("/", 1)[-1]] = body
    for name in ("catvton_engine.py", "jobs.py", "server.py", "mock_engine.py", "requirements-server.txt"):
        assert embedded.get(name) == (ROOT / "tryon_server" / name).read_text(encoding="utf-8"), (
            f"{name} in the notebook is stale: run scripts/build_colab_notebook.py"
        )


def test_all_code_cells_compile():
    for i, source in enumerate(code_cells()):
        if source.startswith("%%writefile"):
            header, body = source.split("\n", 1)
            if header.endswith(".py"):
                compile(body, header, "exec")
            continue
        python = "\n".join("pass" if line.lstrip().startswith(("!", "%")) else line for line in source.splitlines())
        compile(python, f"notebook cell {i}", "exec")
