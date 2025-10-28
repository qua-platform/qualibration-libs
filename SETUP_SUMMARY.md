# Qualibration Libs Docker Setup - Summary

## What Was Created

This Docker setup provides a complete, automated environment for running qualibration-libs tests. Here's what was created:

### Core Files
- **`Dockerfile`** - Defines the Docker image with all dependencies
- **`docker-compose.yml`** - Orchestrates the container with proper volume mounts
- **`auto_setup.py`** - Automated script that handles interactive setup steps
- **`.dockerignore`** - Optimizes Docker build by excluding unnecessary files

### Helper Scripts
- **`run_test.sh`** - Linux/Mac helper script for easy test execution
- **`run_test.bat`** - Windows helper script for easy test execution

### Documentation
- **`DOCKER_README.md`** - Comprehensive usage instructions
- **`SETUP_SUMMARY.md`** - This summary file

## How It Works

1. **Docker Image Build:**
   - Clones qua-libs repository from GitHub
   - Installs qualibration-libs in editable mode
   - Installs superconducting calibrations in editable mode
   - Runs automated setup with default configurations

2. **Volume Mounts:**
   - Current qualibration-libs directory → `/app/qualibration-libs`
   - External data directory → `/app/qualibration_graphs/superconducting/data/QPU_project/2025-10-01`
   - Plots output → `./scripts/plots/`
   - Optional quam_state → `/app/qualibration_graphs/superconducting/quam_state`

3. **Automated Setup:**
   - Handles `setup-qualibrate-config` interactive prompts
   - Copies quam_state from data directories if available
   - Sets up proper Python paths

## Usage Examples

### Quick Start
```bash
# Build and start
docker-compose up --build

# Run all tests
docker-compose exec qualibration-tests python scripts/run_all_tests.py --save

# Or run a specific test
docker-compose exec qualibration-tests python scripts/plot_02a_resonator_spectroscopy_results.py --save
```

### Using Helper Scripts
```bash
# Linux/Mac
./run_test.sh --build
./run_test.sh run_all_tests.py                    # Run all tests
./run_test.sh plot_02a_resonator_spectroscopy_results.py  # Run specific test

# Windows
run_test.bat --build
run_test.bat run_all_tests.py                     # Run all tests
run_test.bat plot_02a_resonator_spectroscopy_results.py   # Run specific test
```

### Custom Data Directory
```bash
# Set custom data location
export DATA_DIR=/path/to/your/data
docker-compose up --build
```

## Key Features

✅ **Fully Automated** - No manual setup steps required  
✅ **Reproducible** - Same environment every time  
✅ **Configurable** - Custom data directories via environment variables  
✅ **Cross-Platform** - Works on Windows, Mac, and Linux  
✅ **Volume Mounts** - Generated plots accessible outside container  
✅ **Error Handling** - Graceful handling of missing dependencies  

## File Structure

```
qualibration-libs/
├── Dockerfile                 # Docker image definition
├── docker-compose.yml        # Container orchestration
├── auto_setup.py             # Automated setup script
├── run_test.sh               # Linux/Mac helper script
├── run_test.bat              # Windows helper script
├── .dockerignore             # Docker build optimization
├── DOCKER_README.md          # Detailed usage instructions
├── SETUP_SUMMARY.md          # This summary
├── scripts/
│   ├── plots/               # Generated plots (mounted externally)
│   │   └── .gitkeep         # Ensures directory is tracked
│   └── *.py                 # Test scripts
└── ...                      # Other qualibration-libs files
```

## Next Steps

1. **Test the setup:**
   ```bash
   docker-compose up --build
   ```

2. **Verify data access:**
   - Ensure your data is in the expected location
   - Check volume mounts are working correctly

3. **Run tests:**
   ```bash
   docker-compose exec qualibration-tests python scripts/plot_02a_resonator_spectroscopy_results.py --save
   ```

4. **Check results:**
   - Look in `scripts/plots/` for generated plots

## Troubleshooting

- **Data not found:** Check `DATA_DIR` environment variable and data structure
- **Permission issues:** Use `sudo chown -R $USER:$USER scripts/plots/` on Linux/Mac
- **Build failures:** Try `docker-compose build --no-cache` to rebuild from scratch
- **Container issues:** Use `docker-compose logs qualibration-tests` to see logs

This setup eliminates the pain of manual environment configuration and makes it easy for anyone to run the qualibration-libs tests with a simple `docker-compose up --build` command!



