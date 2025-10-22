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
from qualibration_libs.plotting.overlays import RefLine, ScatterOverlay  # noqa: E402


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


def plot_resonator_spectroscopy_vs_flux_plotly(
    ds_raw: xr.Dataset, qubits, ds_fit: xr.Dataset, folder_name: str
):
    """Plot 2D heatmap (frequency vs flux) with fit overlay using new Plotly interface."""
    grid = QubitGrid(ds_raw, [q.grid_location for q in qubits])
    qubit_names = [q.name for q in qubits]
    ds_raw = ds_raw.assign_coords(freq_GHz=ds_raw.full_freq / 1e9)
    ds_raw.freq_GHz.attrs["long_name"] = "Frequency [GHz]"
    ds_raw.attenuated_current.attrs["long_name"] = "Current [A]"

    # Create per-qubit overlays
    def create_overlays(qubit_name, qubit_data):
        """Create overlays for each qubit."""
        overlays = []

        try:
            # Get fit data for this qubit
            fit_data = ds_fit.sel(qubit=qubit_name)

            # Add vertical line for idle offset (red dashed)
            idle_offset = float(fit_data.fit_results.idle_offset.values)
            overlays.append(RefLine(x=idle_offset, name="Idle offset", dash="dash"))

            # Add vertical line for flux minimum (orange dashed)
            flux_min = float(fit_data.fit_results.flux_min.values)
            overlays.append(RefLine(x=flux_min, name="Min offset", dash="dash"))

            sweet_spot_freq = fit_data.fit_results.sweet_spot_frequency.values * 1e-9

            overlays.append(
                ScatterOverlay(
                    x=np.array([idle_offset]),
                    y=np.array([sweet_spot_freq]),
                    name="Sweet spot",
                    marker_size=15,
                )
            )

        except Exception as e:
            print(f"Warning: Could not create overlays for {qubit_name}: {e}")

        return overlays

    # Plot 2D heatmap with overlays
    fig = qplot.QualibrationFigure.plot(
        ds_raw,
        x="flux_bias",
        x2="attenuated_current",
        y="freq_GHz",
        data_var="IQ_abs",
        grid=grid,
        qubit_dim="qubit",
        qubit_names=qubit_names,
        overlays=create_overlays,
        showscale=False,
        robust=True,
        title=f"Resonator spectroscopy vs flux - {folder_name}",
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot 02c resonator spectroscopy vs flux results using Plotly."
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
        if p.is_dir() and "02c_resonator_spectroscopy_vs_flux" in p.name
    ]

    if not candidates:
        print("No matching 02c_resonator_spectroscopy_vs_flux result folders found.")
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

        fig = plot_resonator_spectroscopy_vs_flux_plotly(
            ds_raw, qubits, ds_fit, folder.name
        )

        if args.save:
            out_dir = plots_root / folder.name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "res_spectroscopy_vs_flux_plotly.png"

            fig.figure.write_image(str(out_path), width=1500, height=900)
            print(f"Saved: {out_path}")
        else:
            fig.figure.show()


if __name__ == "__main__":
    main()
