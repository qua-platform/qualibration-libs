@echo off
REM Helper script to run qualibration tests in Docker (Windows)

setlocal enabledelayedexpansion

echo 🚀 Qualibration Libs Test Runner
echo ==================================

REM Check if docker-compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ docker-compose not found. Please install Docker Compose.
    exit /b 1
)

REM Parse arguments
set BUILD=false
set UP=false
set DOWN=false
set LOGS=false
set SHELL=false
set SCRIPT_NAME=

:parse_args
if "%~1"=="" goto :end_parse
if "%~1"=="--build" (
    set BUILD=true
    shift
    goto :parse_args
)
if "%~1"=="--up" (
    set UP=true
    shift
    goto :parse_args
)
if "%~1"=="--down" (
    set DOWN=true
    shift
    goto :parse_args
)
if "%~1"=="--logs" (
    set LOGS=true
    shift
    goto :parse_args
)
if "%~1"=="--shell" (
    set SHELL=true
    shift
    goto :parse_args
)
if "%~1"=="--help" (
    call :show_usage
    exit /b 0
)
if "%~1"=="" goto :end_parse
set SCRIPT_NAME=%~1
shift
goto :parse_args

:end_parse

REM Handle different commands
if "%DOWN%"=="true" (
    echo 🛑 Stopping qualibration-tests container...
    docker-compose down
    exit /b 0
)

if "%LOGS%"=="true" (
    echo 📋 Showing logs from qualibration-tests container...
    docker-compose logs -f qualibration-tests
    exit /b 0
)

if "%SHELL%"=="true" (
    echo 🐚 Opening shell in qualibration-tests container...
    docker-compose exec qualibration-tests bash
    exit /b 0
)

if "%UP%"=="true" (
    echo 🚀 Starting qualibration-tests environment in background...
    docker-compose up -d
    echo ✅ Environment started! Use 'run_test.bat --shell' to access the container.
    exit /b 0
)

REM Build if requested
if "%BUILD%"=="true" (
    echo 🔨 Building Docker image...
    docker-compose build
)

REM If no script specified, start interactive environment
if "%SCRIPT_NAME%"=="" (
    echo 🚀 Starting qualibration-tests environment...
    docker-compose up
    exit /b 0
)

REM Validate script name
if not exist "scripts\%SCRIPT_NAME%" (
    echo ❌ Script not found: scripts\%SCRIPT_NAME%
    echo Available scripts:
    dir /b scripts\*.py 2>nul
    exit /b 1
)

REM Run the specific script
echo 🚀 Running %SCRIPT_NAME% in Docker...
echo Command: python scripts\%SCRIPT_NAME% --save
echo.

docker-compose run --rm qualibration-tests python scripts\%SCRIPT_NAME% --save

echo.
echo ✅ Test completed! Check scripts\plots\ for generated plots.
exit /b 0

:show_usage
echo Usage: %0 [OPTIONS] [SCRIPT_NAME]
echo.
echo Options:
echo   --build     Build the Docker image before running
echo   --up        Start the environment in background
echo   --down      Stop the environment
echo   --logs      Show logs from the container
echo   --shell     Open a shell in the container
echo   --help      Show this help message
echo.
echo Available scripts:
echo   run_all_tests.py                                 # Run all plotting tests
echo   plot_02a_resonator_spectroscopy_results.py
echo   plot_02b_resonator_spectroscopy_vs_power_results.py
echo   plot_02c_resonator_spectroscopy_vs_flux_results.py
echo   plot_04b_power_rabi_results.py
echo.
echo Examples:
echo   %0 --build                                    # Build the environment
echo   %0 --up                                       # Start environment in background
echo   %0 run_all_tests.py                           # Run all plotting tests
echo   %0 plot_02a_resonator_spectroscopy_results.py # Run specific test
echo   %0 --shell                                    # Open shell in container
exit /b 0



