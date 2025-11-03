# Qualibration Libs Docker Environment

This Docker setup provides a complete, reproducible environment for **running qualibration-libs tests only**. It is designed for testing and validation purposes, not for general development or production use.

## Intended Use Cases

- ✅ **Testing plotting scripts** with sample data
- ✅ **Validating qualibration-libs functionality** 
- ✅ **Running automated test suites**
- ✅ **Demonstrating plotting capabilities**
- ❌ **General development** (use local installation instead)
- ❌ **Production data processing** (use local installation instead)
- ❌ **Interactive development** (use local installation instead)

## Quick Start

### With Test Data (Recommended)

1. **Place your data file**: Put `preliminary_datasets.zip` in the `qualibration-libs` directory
2. **Build and start the environment:**
   ```bash
   docker-compose up --build
   ```
   The setup will automatically extract and organize your data.

3. **Run all test scripts:**
   ```bash
   docker-compose exec qualibration-tests python scripts/run_all_tests.py
   ```
   
   **Or run a specific test script:**
   ```bash
   docker-compose exec qualibration-tests python scripts/plot_02a_resonator_spectroscopy_results.py --save
   ```

4. **View generated plots:**
   Check the `scripts/plots/` directory for generated plot files.

### Without Test Data

1. **Build and start the environment:**
   ```bash
   docker-compose up --build
   ```

2. **Run all test scripts:**
   ```bash
   docker-compose exec qualibration-tests python scripts/run_all_tests.py --save
   ```
   
   **Or run a specific test script:**
   ```bash
   docker-compose exec qualibration-tests python scripts/plot_02a_resonator_spectroscopy_results.py --save
   ```
   (Note: You'll see "No matching result folders found" without test data)

## Environment Variables

You can customize the data source using environment variables:

```bash
# Set custom data directory (default: ../data)
export DATA_DIR=/path/to/your/data

# Then run docker-compose
docker-compose up --build
```

## Available Test Scripts

- `run_all_tests.py` - Run all plotting tests with --save option
- `plot_02a_resonator_spectroscopy_results.py` - Resonator spectroscopy plots
- `plot_02b_resonator_spectroscopy_vs_power_results.py` - Resonator spectroscopy vs power
- `plot_02c_resonator_spectroscopy_vs_flux_results.py` - Resonator spectroscopy vs flux
- `plot_04b_power_rabi_results.py` - Power Rabi plots

All scripts support the `--save` flag to save plots instead of displaying them.

## Directory Structure

```
qualibration-libs/
├── Dockerfile                 # Docker image definition
├── docker-compose.yml        # Docker Compose configuration
├── auto_setup.py             # Automated setup script
├── DOCKER_README.md          # This file
├── scripts/                  # Test scripts
│   ├── plots/               # Generated plots (mounted externally)
│   └── *.py                 # Individual test scripts
└── ...                      # Other qualibration-libs files
```

## Automatic Data Setup

If you provide a `preliminary_datasets.zip` file in the `qualibration-libs` directory, the setup will automatically:

1. **Extract the zip file** to a temporary location
2. **Find result folders** (folders starting with `#` like `#659_02a_resonator_spectroscopy_182214`)
3. **Copy them to the correct location** (`/app/qualibration_graphs/superconducting/data/QPU_project/2025-10-01/`)
4. **Find quam_state directories** and copy them to the superconducting directory
5. **Organize everything** so plotting scripts can find the data

### Expected Data Structure

Your `preliminary_datasets.zip` should contain:

```
preliminary_datasets.zip
├── #659_02a_resonator_spectroscopy_182214/
│   ├── ds_raw.h5
│   ├── ds_fit.h5
│   └── ... (other result files)
├── #660_02b_resonator_spectroscopy_vs_power_182215/
│   ├── ds_raw.h5
│   ├── ds_fit.h5
│   └── ... (other result files)
├── quam_state/
│   ├── (quam state files)
│   └── ...
└── ... (other result folders)
```

## What the Docker Setup Does

1. **Clones qua-libs repository** from GitHub
2. **Installs qualibration-libs** in editable mode
3. **Installs superconducting calibrations** in editable mode
4. **Runs setup-qualibrate-config** with default settings (automated)
5. **Copies quam_state** from data directories if available
6. **Automatically extracts and organizes preliminary datasets** (if provided)
7. **Mounts external directories** for data and plots

## Troubleshooting

### Data Not Found
If you get "No matching result folders found" errors:
- **With preliminary data**: Ensure your `preliminary_datasets.zip` contains folders starting with `#` (result folders)
- **Without preliminary data**: This is expected - the scripts need actual measurement data to plot
- **Manual data setup**: Copy your zip file into the container:
  ```bash
  docker cp preliminary_datasets.zip qualibration-libs-qualibration-tests-1:/app/preliminary_datasets.zip
  docker-compose restart qualibration-tests
  ```

### Data Structure Issues
- **Missing quam_state**: The script will look for `quam_state` directories in your zip file
- **Wrong folder names**: Result folders must start with `#` (e.g., `#659_02a_resonator_spectroscopy_182214`)
- **Missing data files**: Each result folder should contain `ds_raw.h5` and `ds_fit.h5` files

### Permission Issues
If you encounter permission issues with generated plots:
```bash
# Fix ownership of generated files
sudo chown -R $USER:$USER scripts/plots/
```

### Rebuilding the Environment
To rebuild from scratch:
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```

## Development

**Note**: This Docker environment is for testing only. For actual development work, use a local installation of qualibration-libs.

To modify the test environment:
1. Edit `Dockerfile` for system-level changes
2. Edit `docker-compose.yml` for volume mounts and environment variables
3. Edit `auto_setup.py` for setup automation changes

## Data Locations

- **Result folders**: `/app/qualibration_graphs/superconducting/data/QPU_project/2025-10-01/`
- **quam_state**: `/app/qualibration_graphs/superconducting/quam_state/`
- **Plots output**: `/app/qualibration-libs/scripts/plots/` (mounted to `./scripts/plots/` on host)

## Example Usage

```bash
# Start the environment
docker-compose up --build

# In another terminal, run a test
docker-compose exec qualibration-tests python scripts/plot_02a_resonator_spectroscopy_results.py --save

# Check the results
ls -la scripts/plots/
```

