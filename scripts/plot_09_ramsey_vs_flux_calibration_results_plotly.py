import sys
from pathlib import Path
import argparse
import xarray as xr

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
from qualibration_libs.plotting.overlays import RefLine  # noqa: E402


def extract_node_id_from_folder_name(name: str) -> int | None:
    """Extract numeric id between '#' and first '_' in folder name.

    Example: '#123_09_ramsey_vs_flux_calibration_...' -> 123
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
    base_path = folder.parent.parent
    try:
        loaded = QualibrationNode.load_from_id(node_id=node_id, base_path=base_path)
        if loaded is None:
            return None
        return get_qubits(loaded)
    except Exception:
        return None


def load_dataset(path: Path) -> xr.Dataset:
    """Load an xarray dataset from disk."""
    return xr.load_dataset(path)


def plot_ramsey_vs_flux_raw_plotly(
    ds_raw: xr.Dataset, qubits, ds_fit: xr.Dataset, folder_name: str
):
    """Plot 2D Ramsey state vs idle time and flux using the Plotly interface."""
    grid = QubitGrid(ds_raw, [q.grid_location for q in qubits])
    qubit_names = [q.name for q in qubits]

    ds_plot = ds_raw.copy()
    # State population
    if "state" not in ds_plot:
        raise RuntimeError("Expected 'state' variable in ds_raw for Ramsey vs flux node.")
    ds_plot["state"].attrs["long_name"] = "State population"

    # Time axes: idle_times (ns) and secondary axis in µs
    if "idle_times" not in ds_plot.coords:
        raise RuntimeError("Expected 'idle_times' coordinate in ds_raw.")
    ds_plot.idle_times.attrs["long_name"] = "Idle time [ns]"
    ds_plot = ds_plot.assign_coords(idle_time_us=ds_plot.idle_times / 1e3)
    ds_plot.idle_time_us.attrs["long_name"] = "Idle time [µs]"

    # Flux axis
    if "flux_bias" in ds_plot.coords:
        ds_plot.flux_bias.attrs["long_name"] = "Flux bias [V]"

    # Overlays: horizontal line at the fitted flux_offset for each qubit,
    # matching the matplotlib version (ax.axhline(flux_offset, ...)).
    def create_overlays(qubit_name, qubit_data):
        overlays = []
        try:
            fit_q = ds_fit.sel(qubit=qubit_name)
            flux_offset = float(fit_q.flux_offset.values)
            overlays.append(
                RefLine(
                    y=flux_offset,
                    name="Flux offset",
                    dash="dash",
                )
            )
        except Exception as e:
            print(f"Warning: Could not create raw overlays for {qubit_name}: {e}")
        return overlays

    fig = qplot.QualibrationFigure.plot(
        ds_plot,
        x="idle_times",
        x2="idle_time_us",
        y="flux_bias",
        data_var="state",
        grid=grid,
        qubit_dim="qubit",
        qubit_names=qubit_names,
        overlays=create_overlays,
        showscale=True,
        robust=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.21,
        show_legend=False,
        title=f"Ramsey vs flux (state) - {folder_name}",
        colorbar_tolerance=0.40,
    )

    return fig


def plot_ramsey_vs_flux_parabola_plotly(
    ds_raw: xr.Dataset, qubits, ds_fit: xr.Dataset, folder_name: str
):
    """Plot fitted qubit frequency vs flux (parabolas) using the Plotly interface."""
    grid = QubitGrid(ds_raw, [q.grid_location for q in qubits])
    qubit_names = [q.name for q in qubits]

    # Build the parabola dataset analogous to the matplotlib version:
    # (fit.sel(fit_vals="f").fit_results * 1e3 - detuning)
    if "fit_results" not in ds_fit:
        raise RuntimeError("Expected 'fit_results' variable in ds_fit.")
    if "flux_bias" not in ds_fit.coords:
        raise RuntimeError("Expected 'flux_bias' coordinate in ds_fit.")

    freq_detuning = (
        ds_fit.fit_results.sel(fit_vals="f") * 1e3 - ds_fit.artifitial_detuning
    )
    freq_detuning = freq_detuning.rename("sweetspot_detuning")
    freq_detuning.attrs["long_name"] = "Qubit SweetSpot detuning [MHz]"

    ds_para = xr.Dataset(
        {"sweetspot_detuning": freq_detuning},
        coords={"qubit": ds_fit.qubit, "flux_bias": ds_fit.flux_bias},
    )
    ds_para.flux_bias.attrs["long_name"] = "Flux offset [V]"

    # Overlays: flux_offset (vertical) and detuning from SweetSpot (horizontal)
    def create_overlays(qubit_name, qubit_data):
        overlays = []
        try:
            fit_q = ds_fit.sel(qubit=qubit_name)
            flux_offset = float(fit_q.flux_offset.values)
            detuning = float(fit_q.artifitial_detuning.values)
            freq_offset = float(fit_q.freq_offset.values) * 1e-3 - detuning

            overlays.append(
                RefLine(x=flux_offset, name="Flux offset", dash="dash")
            )
            overlays.append(
                RefLine(
                    y=freq_offset,
                    name="Detuning from SweetSpot",
                    dash="dash",
                )
            )
        except Exception as e:
            print(f"Warning: Could not create overlays for {qubit_name}: {e}")
        return overlays

    fig = qplot.QualibrationFigure.plot(
        ds_para,
        x="flux_bias",
        data_var="sweetspot_detuning",
        grid=grid,
        qubit_dim="qubit",
        qubit_names=qubit_names,
        overlays=create_overlays,
        horizontal_spacing=0.08,
        vertical_spacing=0.21,
        colorbar_tolerance=0.40,
        title=f"Ramsey vs flux (parabola) - {folder_name}",
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot 09 Ramsey vs flux calibration results using Plotly."
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

    # Find all result folders for node 09
    candidates = [
        p
        for p in date_path.iterdir()
        if p.is_dir() and "09_ramsey_vs_flux_calibration" in p.name
    ]

    if not candidates:
        print("No matching 09_ramsey_vs_flux_calibration result folders found.")
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

        fig_raw = plot_ramsey_vs_flux_raw_plotly(ds_raw, qubits, ds_fit, folder.name)
        fig_para = plot_ramsey_vs_flux_parabola_plotly(
            ds_raw, qubits, ds_fit, folder.name
        )

        if args.save:
            out_dir = plots_root / folder.name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_raw = out_dir / "ramsey_vs_flux_raw_plotly.html"
            out_para = out_dir / "ramsey_vs_flux_parabola_plotly.html"

            fig_raw.figure.write_html(str(out_raw))
            fig_para.figure.write_html(str(out_para))

            print(f"Saved: {out_raw}")
            print(f"Saved: {out_para}")
        else:
            fig_raw.figure.show()
            fig_para.figure.show()


if __name__ == "__main__":
    main()


