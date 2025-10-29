import sys
from pathlib import Path
import argparse
from typing import Optional
import xarray as xr

# Paths derived relative to this script's location
scripts_dir = Path(__file__).resolve().parent
repo_root = scripts_dir.parent
SUPERCONDUCTING_DIR = str(repo_root / "qua-libs" / "qualibration_graphs" / "superconducting")
QUALIBRATION_LIBS_DIR = str(repo_root / "qualibration-libs")
DEFAULT_DATE_DIR = "/Users/itayabar/Downloads/preliminary_datasets"

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


def extract_node_id_from_folder_name(name: str) -> Optional[int]:
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
        # Fallback: create simple qubit objects from dataset
        ds_raw_path = folder / "ds_raw.h5"
        if ds_raw_path.exists():
            try:
                ds_raw = xr.open_dataset(ds_raw_path)
                if 'qubit' in ds_raw.coords:
                    qubit_names = list(ds_raw.coords['qubit'].values)
                    # Create simple qubit objects with default grid locations
                    from types import SimpleNamespace
                    qubits = []
                    for i, name in enumerate(qubit_names):
                        qubit = SimpleNamespace()
                        qubit.name = name
                        # Create a simple grid layout
                        qubit.grid_location = f"{i % 2},{i // 2}"  # col,row format
                        qubits.append(qubit)
                    return qubits
            except Exception:
                pass
        return None


def load_dataset(path: Path) -> xr.Dataset:
    # Let xarray auto-detect the engine; h5netcdf is typically used
    return xr.load_dataset(path)


def plot_phase_plotly(ds_raw: xr.Dataset, qubits, folder_name: str):
    """Plot phase data using new Plotly interface."""
    # Create QubitGrid from qubits  
    # Note: grid_location format is "col,row" but QubitGrid expects (row, col), so we swap
    # The old matplotlib code flips rows and normalizes both rows and columns
    coords = {}
    max_row = max(int(q.grid_location.split(',')[1]) for q in qubits)
    min_row = min(int(q.grid_location.split(',')[1]) for q in qubits)
    min_col = min(int(q.grid_location.split(',')[0]) for q in qubits)
    
    for q in qubits:
        col, row = map(int, q.grid_location.split(','))
        # Flip: higher rows (qD=3) → lower positions (top)
        # max_row - row: qD(3)→0, qC(2)→1, so qD appears above qC
        flipped_row = max_row - row
        # Normalize column to start at 0
        normalized_col = col - min_col
        coords[q.name] = (flipped_row, normalized_col)
    
    # Let QubitGrid auto-detect the shape based on the coordinates
    grid = QubitGrid(coords)
    
    # Sort qubit names by their grid position (row, col) to ensure correct plotting order
    sorted_qubit_names = sorted(coords.keys(), key=lambda q: coords[q])
    
    # Plot phase vs detuning
    fig = qplot.QualibrationFigure.plot(
        ds_raw,
        x='detuning',
        data_var='phase',
        grid=grid,
        qubit_dim='qubit',
        qubit_names=sorted_qubit_names,
        title=f"Resonator spectroscopy (phase) - {folder_name}"
    )
    
    return fig


def plot_amplitude_with_fit_plotly(ds_raw: xr.Dataset, qubits, ds_fit: xr.Dataset, folder_name: str):
    """Plot amplitude with fit overlay using new Plotly interface."""
    # Create QubitGrid from qubits
    # Note: grid_location format is "col,row" but QubitGrid expects (row, col), so we swap
    # Also, the old matplotlib code reverses rows and normalizes both rows and columns
    coords = {}
    max_row = max(int(q.grid_location.split(',')[1]) for q in qubits)
    min_row = min(int(q.grid_location.split(',')[1]) for q in qubits)
    min_col = min(int(q.grid_location.split(',')[0]) for q in qubits)
    
    for q in qubits:
        col, row = map(int, q.grid_location.split(','))
        # Reverse row order to match matplotlib layout (higher rows appear at top)
        flipped_row = max_row - row
        # Normalize column to start at 0
        normalized_col = col - min_col
        coords[q.name] = (flipped_row, normalized_col)
    
    # Let QubitGrid auto-detect the shape based on the coordinates
    grid = QubitGrid(coords)
    
    # Sort qubit names by their grid position (row, col) to ensure correct plotting order
    sorted_qubit_names = sorted(coords.keys(), key=lambda q: coords[q])
    
    # Create per-qubit fit overlays
    # We need to re-create the fitted curve here because it is not stored in the dataset
    def create_fit_overlay(qubit_name, qubit_data):
        """Create fit overlay for each qubit."""
        try:
            # Get fit data for this qubit
            fit_data = ds_fit.sel(qubit=qubit_name)
            
            # Check if fit was successful (has valid parameters)
            if 'amplitude' not in fit_data or 'position' not in fit_data or 'width' not in fit_data:
                return []
            
            amplitude = float(fit_data['amplitude'].values)
            position = float(fit_data['position'].values)
            width = float(fit_data['width'].values)
            baseline = float(fit_data['base_line'].mean().values)
            
            # Compute fitted curve using lorentzian_dip
            detuning = ds_raw.coords['detuning'].values
            fitted_curve = lorentzian_dip(detuning, amplitude, position, width / 2, baseline)
            # Convert fit curve to mV to match data
            fitted_curve_mV = fitted_curve * 1000
            
            # Create FitOverlay without text parameters to avoid cluttering the plot
            fit_overlay = FitOverlay(
                y_fit=fitted_curve_mV,
                name="Fit"
            )
            
            return [fit_overlay]
        except Exception as e:
            print(f"Warning: Could not create fit overlay for {qubit_name}: {e}")
            return []
    
    # Convert IQ_abs to mV to match matplotlib plots
    ds_raw_mV = ds_raw.copy()
    ds_raw_mV['IQ_abs'] = ds_raw['IQ_abs'] * 1000  # Convert V to mV
    ds_raw_mV['IQ_abs'].attrs['units'] = 'mV'
    ds_raw_mV['IQ_abs'].attrs['long_name'] = r'$R=\sqrt{I^2 + Q^2}$'
    
    # Plot amplitude vs detuning with fits
    fig = qplot.QualibrationFigure.plot(
        ds_raw_mV,
        x='detuning',
        data_var='IQ_abs',
        grid=grid,
        qubit_dim='qubit',
        qubit_names=sorted_qubit_names,
        overlays=create_fit_overlay,
        title=f"Resonator spectroscopy (amplitude + fit) - {folder_name}"
    )
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot 02a resonator spectroscopy results using Plotly.")
    parser.add_argument("--date-dir", default=DEFAULT_DATE_DIR, help="Path to the date directory to scan.")
    parser.add_argument("--save", action="store_true", help="Save figures to PNG files instead of showing.")
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

