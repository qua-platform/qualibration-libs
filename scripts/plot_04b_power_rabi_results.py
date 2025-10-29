import sys
from pathlib import Path
import argparse
import xarray as xr
import matplotlib.pyplot as plt


# Paths derived relative to this script's location
scripts_dir = Path(__file__).resolve().parent
# Handle both cases: running from scripts/ or from qualibration-libs/
if scripts_dir.name == "scripts":
    repo_root = scripts_dir.parent.parent  # Go up two levels from scripts/
else:
    repo_root = scripts_dir.parent  # Go up one level
SUPERCONDUCTING_DIR = str(repo_root / "qualibration_graphs" / "superconducting")
DEFAULT_DATE_DIR = str(Path(SUPERCONDUCTING_DIR) / "data" / "QPU_project" / "2025-10-01")


# Ensure we can import the plotting utilities exactly like the node
if SUPERCONDUCTING_DIR not in sys.path:
    sys.path.append(SUPERCONDUCTING_DIR)

from calibration_utils.power_rabi.plotting import (  # noqa: E402
    plot_raw_data_with_fit,
)
from qualibrate import QualibrationNode  # noqa: E402
from qualibration_libs.parameters import get_qubits  # noqa: E402


def extract_node_id_from_folder_name(name: str) -> int | None:
    """Extract numeric id between '#' and first '_' in folder name.

    Example: '#123_04b_power_rabi_...' -> 123
    """
    try:
        if not name.startswith("#"):
            return None
        after_hash = name[1:]
        underscore_idx = after_hash.find("_")
        segment = after_hash if underscore_idx == -1 else after_hash[:underscore_idx]
        segment = "".join(ch for ch in segment if ch.isdigit())
        return int(segment) if segment else None
    except Exception:
        return None


def load_qubits_from_folder(folder: Path):
    node_id = extract_node_id_from_folder_name(folder.name)
    if node_id is None:
        return None
    base_path = folder.parent.parent
    try:
        loaded = QualibrationNode.load_from_id(node_id=node_id, base_path=base_path)
        if loaded is None:
            return None
        return get_qubits(loaded)
    except Exception:
        return None


def load_dataset(path: Path) -> xr.Dataset:
    return xr.load_dataset(path)


def main():
    parser = argparse.ArgumentParser(description="Plot 04b power rabi results.")
    parser.add_argument("--date-dir", default=DEFAULT_DATE_DIR, help="Path to the date directory to scan.")
    parser.add_argument("--save", action="store_true", help="Save figures to PNG files instead of showing.")
    args = parser.parse_args()

    date_path = Path(args.date_dir)
    if not date_path.exists():
        raise FileNotFoundError(f"Date directory not found: {args.date_dir}")

    # Find all result folders for node 04b
    candidates = [
        p
        for p in date_path.iterdir()
        if p.is_dir() and "04b_power_rabi" in p.name
    ]

    if not candidates:
        print("No matching 04b_power_rabi result folders found.")
        return

    print("Found result folders:")
    for p in candidates:
        print(f" - {p}")

    plots_root = scripts_dir / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    for folder in candidates:
        ds_raw_path = folder / "ds_raw.h5"
        ds_fit_path = folder / "ds_fit.h5"
        if not ds_raw_path.exists() or not ds_fit_path.exists():
            print(f"Skipping {folder} (missing ds_raw.h5 or ds_fit.h5)")
            continue

        print(f"\nLoading datasets from: {folder}")
        ds_raw = load_dataset(ds_raw_path)
        ds_fit = load_dataset(ds_fit_path)

        qubits = load_qubits_from_folder(folder)
        if qubits is None:
            raise RuntimeError(
                f"Failed to load qubits for folder {folder.name}. "
                "Ensure the folder name encodes the node id (e.g., #123_...) and data is complete."
            )

        fig = plot_raw_data_with_fit(ds_raw, qubits, ds_fit)
        try:
            fig.canvas.manager.set_window_title(f"Power Rabi: {folder.name}")
        except Exception:
            pass

        if args.save:
            out_dir = plots_root / folder.name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "power_rabi.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {out_path}")

    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()


