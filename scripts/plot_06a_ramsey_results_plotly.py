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
from qualibration_libs.plotting.overlays import FitOverlay  # noqa: E402
from qualibration_libs.plotting import config as plot_config  # noqa: E402
from qualibration_libs.analysis import oscillation_decay_exp  # noqa: E402


def extract_node_id_from_folder_name(name: str) -> int | None:
    """Extract numeric id between '#' and first '_' in folder name.

    Example: '#123_06a_ramsey_...' -> 123
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


def plot_ramsey_plotly(
    ds_raw: xr.Dataset, qubits, ds_fit: xr.Dataset, folder_name: str
):
    """Plot Ramsey data with fit overlay using the Plotly interface.

    This produces one subplot per qubit, with two traces per qubit (detuning_signs = ±1),
    and an overlaid fitted curve.
    """
    grid = QubitGrid(ds_raw, [q.grid_location for q in qubits])
    qubit_names = [q.name for q in qubits]

    # Determine whether we are plotting state or I quadrature
    use_state = "state" in ds_raw
    data_var = "state" if use_state else "I"

    # Prepare dataset for plotting:
    # - convert I to mV if needed
    # - flatten detuning_signs into a hue dimension (Plotly will color by this)
    ds_plot = ds_raw.copy()
    if not use_state:
        ds_plot[data_var] = ds_raw[data_var] * 1e3
        ds_plot[data_var].attrs["units"] = "mV"
        ds_plot[data_var].attrs["long_name"] = "Trans. amp. I [mV]"
    else:
        ds_plot[data_var].attrs["long_name"] = "State population"

    ds_plot.idle_time.attrs["long_name"] = "Idle time [ns]"

    # Create per-qubit overlays: one fitted curve per detuning_signs value (Δ = ±)
    def create_overlays(qubit_name, qubit_data):
        overlays = []
        try:
            fit_q = ds_fit.sel(qubit=qubit_name)
            idle = qubit_data.idle_time.values
            palette = plot_config.CURRENT_PALETTE or plot_config.CURRENT_THEME.colorway

            for detuning_sign in fit_q.detuning_signs.values:
                fit_sign = fit_q.sel(detuning_signs=detuning_sign)

                # Extract fit parameters for this detuning sign
                a = float(fit_sign.fit.sel(fit_vals="a").values)
                f = float(fit_sign.fit.sel(fit_vals="f").values)
                phi = float(fit_sign.fit.sel(fit_vals="phi").values)
                offset = float(fit_sign.fit.sel(fit_vals="offset").values)
                decay = float(fit_sign.fit.sel(fit_vals="decay").values)

                # Reconstruct fitted Ramsey curve
                y_fit = oscillation_decay_exp(idle, a, f, phi, offset, decay)
                if not use_state:
                    # Convert to mV for I data to match plotted units
                    y_fit = 1e3 * y_fit

                # Match fit color to the corresponding data trace color:
                # - detuning_signs = -1 -> first hue level -> palette[0]
                # - detuning_signs = +1 -> second hue level -> palette[1]
                if len(palette) >= 2:
                    if int(detuning_sign) < 0:
                        fit_color = palette[0]
                    else:
                        fit_color = palette[1]
                else:
                    fit_color = None

                overlays.append(
                    FitOverlay(
                        y_fit=y_fit,
                        name=f"Fit Δ={int(detuning_sign):+d}",
                        dash="dash",
                        color=fit_color,
                    )
                )
        except Exception as e:
            print(f"Warning: Could not create fit overlay for {qubit_name}: {e}")
        return overlays

    # For plotting, we treat detuning_signs as hue to get two colored traces per qubit
    fig = qplot.QualibrationFigure.plot(
        ds_plot,
        x="idle_time",
        data_var=data_var,
        hue="detuning_signs",
        grid=grid,
        qubit_dim="qubit",
        qubit_names=qubit_names,
        overlays=create_overlays,
        title=f"Ramsey - {folder_name}",
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot 06a Ramsey results using Plotly."
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

    # Find all result folders for node 06a
    candidates = [
        p for p in date_path.iterdir() if p.is_dir() and "06a_ramsey" in p.name
    ]

    if not candidates:
        print("No matching 06a_ramsey result folders found.")
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

        fig = plot_ramsey_plotly(ds_raw, qubits, ds_fit, folder.name)

        if args.save:
            out_dir = plots_root / folder.name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "ramsey_plotly.html"
            fig.figure.write_html(str(out_path))
            print(f"Saved: {out_path}")
        else:
            fig.figure.show()


if __name__ == "__main__":
    main()
