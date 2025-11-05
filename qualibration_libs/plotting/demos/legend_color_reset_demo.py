import numpy as np
import xarray as xr
import sys
from pathlib import Path

# Ensure repo root (which contains `qualibration_libs/`) is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import qualibration_libs.plotting as qplot
from qualibration_libs.plotting import QubitGrid
from qualibration_libs.plotting.overlays import FitOverlay


def generate_ramsey_data():
    """Generate sample Ramsey experiment data for 6 qubits."""

    qubits = ['qD1', 'qD3', 'qC4', 'qC3', 'qC2', 'qC1']
    detuning_signs = [-1, 1]  # Δ = - and Δ = +
    idle_times = np.linspace(0, 30000, 100)  # 0 to 30k ns

    # Generate oscillating data
    state_data = np.zeros((len(qubits), len(detuning_signs), len(idle_times)))

    for i, qubit in enumerate(qubits):
        for j, detuning_sign in enumerate(detuning_signs):
            base_freq = 0.00008 + i * 0.00001
            decay_rate = 0.00003 + i * 0.000005
            phase_offset = i * 0.2 + j * 0.3
            amplitude = 0.4 + i * 0.05

            for k, t in enumerate(idle_times):
                oscillation = np.sin(2 * np.pi * base_freq * t + phase_offset)
                decay = np.exp(-decay_rate * t)
                noise = 0.01 * np.random.randn()

                state = 0.5 + amplitude * oscillation * decay + noise
                state = np.clip(state, 0, 1)
                state_data[i, j, k] = state

    # Create dataset
    ds = xr.Dataset(
        {'state': (['qubit', 'detuning_signs', 'idle_time'], state_data)},
        coords={'qubit': qubits, 'detuning_signs': detuning_signs, 'idle_time': idle_times}
    )

    return ds


def create_fit_overlays(ds):
    """Create fit overlays for each qubit and detuning sign."""

    def create_qubit_fits(qubit_name, qubit_data):
        overlays = []
        qubit_ds = ds.sel(qubit=qubit_name)
        idle_times = qubit_ds.coords['idle_time'].values

        for detuning_sign in [-1, 1]:
            state_data = qubit_ds.sel(detuning_signs=detuning_sign)['state'].values

            try:
                poly_coeffs = np.polyfit(idle_times, state_data, 4)
                fit_curve = np.polyval(poly_coeffs, idle_times)
            except:
                fit_curve = np.linspace(state_data[0], state_data[-1], len(idle_times))

            fit_overlay = FitOverlay(
                y_fit=fit_curve,
                name=f"Fit Δ={detuning_sign:+d}",
                dash="dash"
            )
            overlays.append(fit_overlay)

        return overlays

    return create_qubit_fits


def main():
    """Create the plot with distinct but consistent colors."""

    print("Simple Solution: Distinct but Consistent Colors")
    print("=" * 50)

    # Generate data
    ds = generate_ramsey_data()

    # Create grid
    qubit_positions = {
        'qD1': (0, 0), 'qD3': (0, 1),
        'qC4': (1, 0), 'qC3': (1, 1),
        'qC2': (2, 0), 'qC1': (2, 1)
    }
    grid = QubitGrid(qubit_positions, shape=(3, 2))

    # Create fit overlays
    fit_overlays = create_fit_overlays(ds)

    # THE KEY SOLUTION: Set a custom 2-color palette
    print("Setting custom 2-color palette...")
    qplot.set_palette(["#1f77b4", "#d62728"])  # Blue, Red

    print("Color mapping:")
    print("  - Delta = - (detuning_signs = -1): Blue")
    print("  - Delta = + (detuning_signs = +1): Red")

    # Create the plot
    fig = qplot.QualibrationFigure.plot(
        ds,
        x="idle_time",
        data_var="state",
        title="Ramsey vs flux - Consistent Colors",
        hue="detuning_signs",
        overlays=fit_overlays,
        grid=grid,
        marker_size=3,
        line_width=1.5
    )

    print("Displaying plot...")
    fig.figure.show()

    print("\nSUCCESS: Now you have:")
    print("  - Delta = - and Delta = + are DISTINCT (Blue vs Red)")
    print("  - Colors are CONSISTENT across all 6 subplots")
    print("  - Fit overlays match their data colors")

    # Reset palette
    qplot.set_palette("qualibrate")


if __name__ == "__main__":
    main()