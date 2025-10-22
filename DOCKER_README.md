# Qualibration Libs Docker Environment

This Docker setup provides a complete, reproducible environment for running qualibration-libs tests without manual setup.

## Quick Start

1. **Build and start the environment:**
   ```bash
   docker-compose up --build
   ```

2. **Run a test script:**
   ```bash
   docker-compose exec qualibration-tests python scripts/plot_02a_resonator_spectroscopy_results.py --save
   ```

3. **View generated plots:**
   Check the `scripts/plots/` directory for generated plot files.

## Environment Variables

You can customize the data source and quam_state location using environment variables:

```bash
# Set custom data directory (default: ../data)
export DATA_DIR=/path/to/your/data

# Set custom quam_state directory (default: ../quam_state)
export QUAM_STATE_DIR=/path/to/your/quam_state

# Then run docker-compose
docker-compose up --build
```

## Available Test Scripts

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

## What the Docker Setup Does

1. **Clones qua-libs repository** from GitHub
2. **Installs qualibration-libs** in editable mode
3. **Installs superconducting calibrations** in editable mode
4. **Runs setup-qualibrate-config** with default settings (automated)
5. **Copies quam_state** from data directories if available
6. **Mounts external directories** for data and plots

## Troubleshooting

### Data Not Found
If you get "Date directory not found" errors:
- Ensure your data is in the correct location
- Check the `DATA_DIR` environment variable
- Verify the data structure matches expected format

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

To modify the environment:
1. Edit `Dockerfile` for system-level changes
2. Edit `docker-compose.yml` for volume mounts and environment variables
3. Edit `auto_setup.py` for setup automation changes

## Example Usage

```bash
# Start the environment
docker-compose up --build

# In another terminal, run a test
docker-compose exec qualibration-tests python scripts/plot_02a_resonator_spectroscopy_results.py --save

# Check the results
ls -la scripts/plots/
```

