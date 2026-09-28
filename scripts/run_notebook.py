"""Execute a plain-Python notebook (no IPython magics) top-to-bottom, headless.

Usage: python3 scripts/run_notebook.py notebooks/01_data_exploration.ipynb
Cells run in one shared namespace with cwd set to the notebook's directory.
Stdout/stderr of each cell are captured into the cell outputs; the notebook
is written back in place. Exits non-zero on the first failing cell.
"""
import contextlib
import io
import os
import sys
import time
import traceback

import nbformat
from nbformat.v4 import new_output

path = os.path.abspath(sys.argv[1])
nb = nbformat.read(path, as_version=4)
nb_dir = os.path.dirname(os.path.abspath(path))
os.chdir(nb_dir)

ns = {"__name__": "__main__"}
t0 = time.time()
n_code = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    n_code += 1
    src = "\n".join(
        ln for ln in cell.source.splitlines()
        if not ln.lstrip().startswith(("%", "!"))
    )
    out_buf, err_buf = io.StringIO(), io.StringIO()
    try:
        code = compile(src, f"<cell-{i}>", "exec")
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            exec(code, ns)
    except Exception:
        traceback.print_exc()
        print(f"\nFAILED at cell index {i} (code cell #{n_code})", file=sys.stderr)
        sys.exit(1)
    outs = []
    if out_buf.getvalue():
        outs.append(new_output("stream", name="stdout", text=out_buf.getvalue()))
    if err_buf.getvalue():
        outs.append(new_output("stream", name="stderr", text=err_buf.getvalue()))
    cell["outputs"] = outs
    cell["execution_count"] = n_code
    # flush progress for long cells
    print(f"[cell {i}] ok ({time.time()-t0:.0f}s elapsed)", flush=True)

nbformat.write(nb, path)
print(f"DONE: {n_code} code cells executed, notebook written back ({time.time()-t0:.0f}s total)")
