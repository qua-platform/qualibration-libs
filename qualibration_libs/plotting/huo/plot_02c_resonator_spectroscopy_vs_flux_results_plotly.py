import sys
from pathlib import Path
import argparse
from typing import Optional
import xarray as xr
import numpy as np

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
from qualibration_libs.plotting.overlays import RefLine, ScatterOverlay  # noqa: E402


def extract_node_id_from_folder_name(name: str) -> Optional[int]:
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
    return xr.load_dataset(path)


def plot_resonator_spectroscopy_vs_flux_plotly(ds_raw: xr.Dataset, qubits, ds_fit: xr.Dataset, folder_name: str):
    """Plot 2D heatmap (frequency vs flux) with fit overlay using new Plotly interface."""
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
    
    # Create per-qubit overlays
    def create_overlays(qubit_name, qubit_data):
        """Create overlays for each qubit."""
        overlays = []
        
        try:
            # Get fit data for this qubit
            fit_data = ds_fit.sel(qubit=qubit_name)
            
            # Check if fit was successful
            if 'fit_results' not in fit_data:
                return overlays
            
            fit_results = fit_data['fit_results']
            if 'success' not in fit_results or not fit_results['success'].values:
                return overlays
            
            # Add vertical line for idle offset (red dashed)
            if 'idle_offset' in fit_results:
                idle_offset = float(fit_results['idle_offset'].values)
                overlays.append(
                    RefLine(
                        x=idle_offset,
                        name="Idle offset",
                        dash="dash"
                    )
                )
            
            # Add vertical line for flux minimum (orange dashed)
            if 'flux_min' in fit_results:
                flux_min = float(fit_results['flux_min'].values)
                overlays.append(
                    RefLine(
                        x=flux_min,
                        name="Min offset",
                        dash="dash"
                    )
                )
            
            # Add scatter point for sweet spot frequency (red star)
            if 'idle_offset' in fit_results and 'sweet_spot_frequency' in fit_results:
                idle_offset = float(fit_results['idle_offset'].values)
                sweet_spot_freq = float(fit_results['sweet_spot_frequency'].values) * 1e-9  # Convert to GHz
                
                overlays.append(
                    ScatterOverlay(
                        x=np.array([idle_offset]),
                        y=np.array([sweet_spot_freq]),
                        name="Sweet spot",
                        marker_size=15
                    )
                )
                
        except Exception as e:
            print(f"Warning: Could not create overlays for {qubit_name}: {e}")
        
        return overlays
    
    # Plot 2D heatmap with overlays
    # Use individual colormaps to avoid overlapping colorbars
    fig = qplot.QualibrationFigure.plot(
        ds_raw,
        x='flux_bias',
        y='full_freq',
        data_var='IQ_abs',
        grid=grid,
        qubit_dim='qubit',
        qubit_names=sorted_qubit_names,
        overlays=create_overlays,
        title=f"Resonator spectroscopy vs flux - {folder_name}",
        colorbar={'title': 'IQ Amplitude (mV)'},  # Smart colorbar logic
        colorbar_tolerance=3.0  # 300% tolerance for testing - always show colorbars
    )
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot 02c resonator spectroscopy vs flux results using Plotly.")
    parser.add_argument("--date-dir", default=DEFAULT_DATE_DIR, help="Path to the date directory to scan.")
    parser.add_argument("--save", action="store_true", help="Save figures to PNG files instead of showing.")
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

        fig = plot_resonator_spectroscopy_vs_flux_plotly(ds_raw, qubits, ds_fit, folder.name)

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

