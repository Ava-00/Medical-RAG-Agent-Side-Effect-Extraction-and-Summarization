import argparse
import sys
import subprocess
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def ensure_deps():
    # Install deps into current interpreter env (Colab)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "nbclient", "nbformat"],
        check=True,
    )


def inject_params_first_cell(nb, params: dict) -> None:
    # Safe param injection (string values only for this use case)
    # We set variables like: EXTERNAL_QUERY = """..."""
    lines = []
    for k, v in params.items():
        if not isinstance(v, str):
            raise TypeError(f"Param {k} must be a string for this runner. Got {type(v)}")
        lines.append(f'{k} = """{v}"""')
    nb.cells.insert(0, nbformat.v4.new_code_cell("\n".join(lines)))


def run_notebook(nb_path: Path, out_path: Path, params: dict | None = None, kernel_name: str = "python3") -> Path:
    nb = nbformat.read(str(nb_path), as_version=4)

    if params:
        inject_params_first_cell(nb, params)

    client = NotebookClient(
        nb,
        timeout=None,  # no per-cell timeout
        kernel_name=kernel_name,
        allow_errors=False,
        resources={"metadata": {"path": str(nb_path.parent)}},  # set working dir to notebook folder
    )

    try:
        client.execute()
    except CellExecutionError as e:
        # Save partial outputs for debugging
        nbformat.write(nb, str(out_path))
        raise RuntimeError(
            f"Notebook failed: {nb_path}\n"
            f"Partial executed notebook saved to: {out_path}\n\n"
            f"{e}"
        ) from e

    nbformat.write(nb, str(out_path))
    return out_path


def main():
    # Delay-drive-mount logic: this script assumes it's being executed in Colab
    # where Drive is available at /content/drive after mounting.
    # We'll mount automatically if possible.
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
    except Exception:
        # If not in Colab, proceed (user might be running locally)
        pass

    ensure_deps()

    ap = argparse.ArgumentParser(description="Execute 3-notebook pipeline with prompt injection for Notebook 3.")
    ap.add_argument(
        "--base",
        required=True,
        help="Folder containing the notebooks (Drive path in Colab). Example: /content/drive/MyDrive/medical-rag-agent",
    )
    ap.add_argument(
        "--prompt",
        required=True,
        help="Prompt to inject into langgraph_workflow_module.ipynb as EXTERNAL_QUERY.",
    )
    ap.add_argument(
        "--outdir",
        default="executed_notebooks",
        help="Name of output subfolder under --base to save executed notebooks.",
    )
    ap.add_argument(
        "--kernel",
        default="python3",
        help='Kernel name (default: "python3").',
    )
    args = ap.parse_args()

    base = Path(args.base).expanduser().resolve()
    nb1 = base / "chunking_module.ipynb"
    nb2 = base / "embeddings_module.ipynb"
    nb3 = base / "langgraph_workflow_module.ipynb"

    out_dir = base / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [p for p in (nb1, nb2, nb3) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "These notebooks were not found (check --base path):\n" + "\n".join(str(m) for m in missing)
        )

    print("\n=== workflow_execution_script ===")
    print("Base folder :", base)
    print("Output dir  :", out_dir)
    print("Prompt      :", args.prompt)
    print("Kernel      :", args.kernel)
    print("Notebooks   :")
    print("  1)", nb1.name)
    print("  2)", nb2.name)
    print("  3)", nb3.name, "(EXTERNAL_QUERY injected)")

    print("\n--- Running 1/3:", nb1.name, "---")
    out1 = run_notebook(nb1, out_dir / "chunking_module.executed.ipynb", kernel_name=args.kernel)
    print("Saved:", out1)

    print("\n--- Running 2/3:", nb2.name, "---")
    out2 = run_notebook(nb2, out_dir / "embeddings_module.executed.ipynb", kernel_name=args.kernel)
    print("Saved:", out2)

    print("\n--- Running 3/3:", nb3.name, "---")
    out3 = run_notebook(
        nb3,
        out_dir / "langgraph_workflow_module.executed.ipynb",
        params={"EXTERNAL_QUERY": args.prompt},
        kernel_name=args.kernel,
    )
    print("Saved:", out3)

    print("\n Pipeline completed successfully.")
    print("Executed notebooks saved under:", out_dir)


if __name__ == "__main__":
    main()
