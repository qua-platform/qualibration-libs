import sys
from pathlib import Path
import argparse
import xarray as xr
import numpy as np

# Paths derived relative to this script's location
scripts_dir = Path(__file__).resolve().parent
# Handle both cases: running from scripts/ or from qualibration-libs/
if scripts_dir.name == "scripts":
    repo_root = scripts_dir.parent.parent  # Go up two levels from scripts/
else:
    repo_root = scripts_dir.parent  # Go up one level
SUPERCONDUCTING_DIR = str(
    repo_root / "qualibration_graphs" / "superconducting"
)
QUALIBRATION_LIBS_DIR = str(repo_root / "qualibration-libs")
DEFAULT_DATE_DIR = str(
    Path(SUPERCONDUCTING_DIR) / "data" / "QPU_project" / "2025-10-01"
)

# Add paths for imports
if SUPERCONDUCTING_DIR not in sys.path:
    sys.path.append(SUPERCONDUCTING_DIR)
if QUALIBRATION_LIBS_DIR not in sys.path:
    sys.path.append(QUALIBRATION_LIBS_DIR)

from qualibrate import QualibrationNode  # noqa: E402
from qualibration_libs.parameters import get_qubits  # noqa: E402
import qualibration_libs.plotting as qplot  # noqa: E402
from qualibration_libs.plotting import QubitGrid  # noqa: E402
from qualibration_libs.plotting.overlays import LineOverlay, RefLine  # noqa: E402


def extract_node_id_from_folder_name(name: str) -> int | None:
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


def plot_resonator_spectroscopy_vs_power_plotly(
    ds_raw: xr.Dataset, qubits, ds_fit: xr.Dataset, folder_name: str
):
    """Plot 2D heatmap (frequency vs power) with fit overlay using new Plotly interface."""
    grid = QubitGrid(ds_raw, [q.grid_location for q in qubits])
    qubit_names = [q.name for q in qubits]

    def create_overlays(qubit_name, qubit_data):
        """Create overlays for each qubit."""
        overlays = []
        try:
            fit_data = ds_fit.sel(qubit=qubit_name)

            # Resonance frequency line
            if "rr_min_response" in fit_data and "power" in fit_data:
                overlays.append(
                    LineOverlay(
                        x=fit_data["rr_min_response"].values * 1e-6,
                        y=fit_data["power"].values,
                        name="Resonance freq",
                        dash="solid",
                        width=1.5,
                    )
                )

            # Optimal power reference lines
            if fit_data["success"].values:
                overlays.append(
                    RefLine(
                        x=float(fit_data["freq_shift"].values) * 1e-6 if "freq_shift" in fit_data else None,
                        y=float(fit_data["optimal_power"].values) if "optimal_power" in fit_data else None,
                        name="Optimal Power",
                        dash="dash",
                        color="red",
                    )
                )
        except Exception as e:
            print(f"Warning: Could not create overlays for {qubit_name}: {e}")

        return overlays

    # Prepare dataset with converted coordinates
    ds_raw_plot = ds_raw.assign_coords(
        detuning_MHz=ds_raw["detuning"] / 1e6,
        freq_GHz=ds_raw.full_freq / 1e9
    )
    ds_raw_plot.freq_GHz.attrs["long_name"] = "Frequency [GHz]"
    ds_raw_plot.detuning_MHz.attrs["long_name"] = "Detuning [MHz]"

    fig = qplot.QualibrationFigure.plot(
        ds_raw_plot,
        x2="freq_GHz",
        x="detuning_MHz",
        y="power",
        data_var="IQ_abs_norm",
        grid=grid,
        qubit_dim="qubit",
        qubit_names=qubit_names,
        robust=True,
        showscale=False,
        overlays=create_overlays,
        title=f"Resonator spectroscopy vs power - {folder_name}",
    )
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot 02b resonator spectroscopy vs power results using Plotly."
    )
    parser.add_argument(
        "--date-dir",
        default=DEFAULT_DATE_DIR,
        help="Path to the date directory to scan.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figures to PNG files instead of showing.",
    )
    args = parser.parse_args()

    date_path = Path(args.date_dir)
    if not date_path.exists():
        raise FileNotFoundError(f"Date directory not found: {args.date_dir}")

    candidates = [
        p
        for p in date_path.iterdir()
        if p.is_dir() and "02b_resonator_spectroscopy_vs_power" in p.name
    ]

    if not candidates:
        print("No matching 02b_resonator_spectroscopy_vs_power result folders found.")
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

        fig = plot_resonator_spectroscopy_vs_power_plotly(
            ds_raw, qubits, ds_fit, folder.name
        )

        if args.save:
            out_dir = plots_root / folder.name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "res_spectroscopy_vs_power_plotly.html"

            fig.figure.write_html(str(out_path))
            print(f"Saved: {out_path}")
        else:
            fig.figure.show()


if __name__ == "__main__":
    main()
