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
from qualibration_libs.plotting.overlays import FitOverlay, RefLine  # noqa: E402
from qualibration_libs.analysis import oscillation  # noqa: E402


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


def plot_power_rabi_1d_plotly(
    ds_raw: xr.Dataset, qubits, ds_fit: xr.Dataset, folder_name: str
):
    """Plot 1D power rabi with fit overlay using new Plotly interface."""
    grid = QubitGrid(ds_raw, [q.grid_location for q in qubits])
    qubit_names = [q.name for q in qubits]

    # Determine which data variable to use (I or state)
    if "I" in ds_raw:
        data_var = "I"
    elif "state" in ds_raw:
        data_var = "state"
    else:
        raise RuntimeError(
            "The dataset must contain either 'I' or 'state' for the plotting function to work."
        )

    # For 1D case, select the single nb_of_pulses value to reduce dimensionality
    ds_raw_1d = ds_raw.sel(nb_of_pulses=ds_raw.nb_of_pulses[0])

    # Create per-qubit fit overlays
    def create_fit_overlay(qubit_name, qubit_data):
        """Create fit overlay for each qubit."""
        overlays = []

        try:
            # Get fit data for this qubit
            fit_data = ds_fit.sel(qubit=qubit_name)

            if "fit" not in fit_data or "amp_prefactor" not in fit_data:
                return overlays

            # Extract fit parameters
            a = float(fit_data["fit"].sel(fit_vals="a").values)
            f = float(fit_data["fit"].sel(fit_vals="f").values)
            phi = float(fit_data["fit"].sel(fit_vals="phi").values)
            offset = float(fit_data["fit"].sel(fit_vals="offset").values)
            amp_prefactor = fit_data["amp_prefactor"].values

            # Compute fitted curve using oscillation function
            fitted_curve = oscillation(amp_prefactor, a, f, phi, offset)
            # Convert to mV only if using I quadrature
            if data_var == "I":
                fitted_curve = fitted_curve * 1e3

            # Create FitOverlay without params to avoid cluttering the plot
            fit_overlay = FitOverlay(y_fit=fitted_curve, name="Fit")

            overlays.append(fit_overlay)

        except Exception as e:
            print(f"Warning: Could not create fit overlay for {qubit_name}: {e}")

        return overlays

    # Convert data to mV to match matplotlib plots (if using I quadrature)
    ds_raw_mV = ds_raw_1d.copy()
    if data_var == "I":
        ds_raw_mV[data_var] = ds_raw_1d[data_var] * 1000  # Convert V to mV
        ds_raw_mV[data_var].attrs["units"] = "mV"
        ds_raw_mV[data_var].attrs["long_name"] = "Rotated I quadrature"

    ds_raw_mV = ds_raw_mV.assign_coords(amp_mV=ds_raw_mV.full_amp * 1000)
    ds_raw_mV.amp_mV.attrs["long_name"] = "Pulse amplitude [mV]"
    ds_raw_mV.amp_prefactor.attrs["long_name"] = "amplitude prefactor"
    # For 'state', no conversion needed as it's already dimensionless

    # Plot 1D power rabi with fit
    fig = qplot.QualibrationFigure.plot(
        ds_raw_mV,
        x="amp_prefactor",
        x2="amp_mV",
        data_var=data_var,
        grid=grid,
        qubit_dim="qubit",
        qubit_names=qubit_names,
        overlays=create_fit_overlay,
        title=f"Power Rabi (1D) - {folder_name}",
    )


    return fig


def plot_power_rabi_2d_plotly(
    ds_raw: xr.Dataset, qubits, ds_fit: xr.Dataset, folder_name: str
):
    """Plot 2D power rabi with optimal amplitude marker using new Plotly interface."""
    grid = QubitGrid(ds_raw, [q.grid_location for q in qubits])
    qubit_names = [q.name for q in qubits]

    # Determine which data variable to use (I or state)
    if "I" in ds_raw:
        data_var = "I"
    elif "state" in ds_raw:
        data_var = "state"
    else:
        raise RuntimeError(
            "The dataset must contain either 'I' or 'state' for the plotting function to work."
        )

    # Create per-qubit overlays
    def create_overlays(qubit_name, qubit_data):
        """Create overlays for each qubit."""
        overlays = []

        try:
            # Get fit data for this qubit
            fit_data = ds_fit.sel(qubit=qubit_name)

            # Add vertical line for optimal amplitude (green)
            if "opt_amp_prefactor" in fit_data and fit_data["success"].values:
                opt_amp_prefactor = float(fit_data["opt_amp_prefactor"].values)
                overlays.append(
                    RefLine(x=opt_amp_prefactor, name="Optimal amplitude", dash="solid")
                )

        except Exception as e:
            print(f"Warning: Could not create overlays for {qubit_name}: {e}")

        return overlays

    # Convert data to mV to match matplotlib plots (if using I quadrature)
    ds_raw_mV = ds_raw.copy()
    if data_var == "I":
        ds_raw_mV[data_var] = ds_raw[data_var] * 1000  # Convert V to mV
        ds_raw_mV[data_var].attrs["units"] = "mV"
        ds_raw_mV[data_var].attrs["long_name"] = "Rotated I quadrature"
    # For 'state', no conversion needed as it's already dimensionless
    ds_raw_mV = ds_raw_mV.assign_coords(amp_mV=ds_raw_mV.full_amp * 1000)
    ds_raw_mV.amp_mV.attrs["long_name"] = "Pulse amplitude [mV]"
    ds_raw_mV.amp_prefactor.attrs["long_name"] = "amplitude prefactor"

    # Plot 2D heatmap with overlays
    fig = qplot.QualibrationFigure.plot(
        ds_raw_mV,
        x="amp_prefactor",
        x2="amp_mV",
        y="nb_of_pulses",
        data_var=data_var,
        grid=grid,
        qubit_dim="qubit",
        qubit_names=qubit_names,
        overlays=create_overlays,
        title=f"Power Rabi (2D) - {folder_name}",
    )


    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot 04b power rabi results using Plotly."
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

    # Find all result folders for node 04b
    candidates = [
        p for p in date_path.iterdir() if p.is_dir() and "04b_power_rabi" in p.name
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

        # Determine if 1D or 2D based on nb_of_pulses dimension
        if len(ds_raw.nb_of_pulses) == 1:
            # 1D case
            fig = plot_power_rabi_1d_plotly(ds_raw, qubits, ds_fit, folder.name)
        else:
            # 2D case
            fig = plot_power_rabi_2d_plotly(ds_raw, qubits, ds_fit, folder.name)

        if args.save:
            out_dir = plots_root / folder.name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "power_rabi_plotly.png"

            fig.figure.write_image(str(out_path), width=1500, height=900)
            print(f"Saved: {out_path}")
        else:
            fig.figure.show()


if __name__ == "__main__":
    main()
