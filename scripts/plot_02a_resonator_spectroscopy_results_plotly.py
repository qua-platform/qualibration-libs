import sys
from pathlib import Path
import argparse
import xarray as xr
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

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
from qualibration_libs.plotting.overlays import FitOverlay  # noqa: E402
from qualibration_libs.analysis import lorentzian_dip  # noqa: E402


def extract_node_id_from_folder_name(name: str) -> int | None:
    """Extract numeric id between '#' and first '_' in folder name.

    Example: '#659_02a_resonator_spectroscopy_182214' -> 659
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
    """Load the original qubits (with grid locations) using QualibrationNode.load_from_id."""
    node_id = extract_node_id_from_folder_name(folder.name)
    if node_id is None:
        return None
    # Base path is the storage root containing date folders
    base_path = folder.parent.parent
    try:
        loaded = QualibrationNode.load_from_id(node_id=node_id, base_path=base_path)
        if loaded is None:
            return None
        # Reconstruct qubits exactly like the node's load_data action
        qubits = get_qubits(loaded)
        return qubits
    except Exception:
        return None


def load_dataset(path: Path) -> xr.Dataset:
    # Let xarray auto-detect the engine; h5netcdf is typically used
    return xr.load_dataset(path)


def plot_phase_plotly(ds_raw: xr.Dataset, qubits, folder_name: str):
    """Plot phase data using new Plotly interface."""
    grid = QubitGrid(ds_raw, [q.grid_location for q in qubits])
    qubit_names = [q.name for q in qubits]
    ds_raw = ds_raw.assign_coords(full_freq_GHz=ds_raw.full_freq / u.GHz)
    ds_raw.full_freq_GHz.attrs["long_name"] = "RF frequency [GHz]"
    # Plot phase vs detuning
    fig = qplot.QualibrationFigure.plot(
        ds_raw,
        x="detuning",
        x2="full_freq_GHz",
        data_var="phase",
        grid=grid,
        qubit_dim="qubit",
        qubit_names=qubit_names,
        title=f"Resonator spectroscopy (phase) - {folder_name}",
    )

    return fig


def plot_amplitude_with_fit_plotly(
    ds_raw: xr.Dataset, qubits, ds_fit: xr.Dataset, folder_name: str
):
    """Plot amplitude with fit overlay using new Plotly interface."""
    grid = QubitGrid(ds_raw, [q.grid_location for q in qubits])
    qubit_names = [q.name for q in qubits]

    # Create per-qubit fit overlays
    # We need to re-create the fitted curve here because it is not stored in the dataset
    def create_fit_overlay(qubit_name, qubit_data):
        """Create fit overlay for each qubit."""
        try:
            # Get fit data for this qubit
            fit_data = ds_fit.sel(qubit=qubit_name)
            fitted_curve = lorentzian_dip(
                ds_raw.detuning,
                float(fit_data.amplitude.values),
                float(fit_data.position.values),
                float(fit_data.width.values) / 2,
                float(fit_data.base_line.mean().values),
            )

            # Convert fit curve to mV to match data
            fitted_curve_mV = fitted_curve / u.mV

            # Create FitOverlay without text parameters to avoid cluttering the plot
            fit_overlay = FitOverlay(y_fit=fitted_curve_mV, name="Fit")

            return [fit_overlay]
        except Exception as e:
            print(f"Warning: Could not create fit overlay for {qubit_name}: {e}")
            return []

    # Convert IQ_abs to mV to match matplotlib plots
    ds_raw_mV = ds_raw.copy()
    ds_raw_mV["IQ_abs"] = ds_raw["IQ_abs"] / u.mV  # Convert V to mV
    ds_raw_mV["IQ_abs"].attrs["units"] = "mV"
    ds_raw_mV["IQ_abs"].attrs["long_name"] = r"$R=\sqrt{I^2 + Q^2}$"
    ds_raw_mV = ds_raw_mV.assign_coords(full_freq_GHz=ds_raw_mV.full_freq / u.GHz)
    ds_raw_mV.full_freq_GHz.attrs["long_name"] = "RF frequency [GHz]"

    # Plot amplitude vs detuning with fits
    fig = qplot.QualibrationFigure.plot(
        ds_raw_mV,
        x="detuning",
        x2="full_freq_GHz",
        data_var="IQ_abs",
        grid=grid,
        qubit_dim="qubit",
        qubit_names=qubit_names,
        overlays=create_fit_overlay,
        title=f"Resonator spectroscopy (amplitude + fit) - {folder_name}",
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot 02a resonator spectroscopy results using Plotly."
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

    # Find all result folders for node 02a
    candidates = [
        p
        for p in date_path.iterdir()
        if p.is_dir() and "02a_resonator_spectroscopy" in p.name
    ]

    if not candidates:
        print("No matching 02a_resonator_spectroscopy result folders found.")
        return

    print("Found result folders:")
    for p in candidates:
        print(f" - {p}")

    scripts_dir = Path(__file__).resolve().parent
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

        # Reconstruct original qubits (with grid locations) from stored node id
        qubits = load_qubits_from_folder(folder)
        if qubits is None:
            raise RuntimeError(
                f"Failed to load qubits for folder {folder.name}. "
                "Ensure the folder name encodes the node id (e.g., #659_...) and data is complete."
            )

        # Plot phase
        fig_phase = plot_phase_plotly(ds_raw, qubits, folder.name)

        # Plot amplitude with fit
        fig_amp = plot_amplitude_with_fit_plotly(ds_raw, qubits, ds_fit, folder.name)

        if args.save:
            # Save to scripts/plots/<data-folder-name>/
            out_dir = plots_root / folder.name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_phase = out_dir / "phase_plotly.png"
            out_amp = out_dir / "amplitude_fit_plotly.png"

            # Save Plotly figures as PNG
            fig_phase.figure.write_image(str(out_phase), width=1500, height=900)
            fig_amp.figure.write_image(str(out_amp), width=1500, height=900)

            print(f"Saved: {out_phase}")
            print(f"Saved: {out_amp}")
        else:
            # Show figures
            fig_phase.figure.show()
            fig_amp.figure.show()


if __name__ == "__main__":
    main()
