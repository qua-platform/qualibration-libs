#!/bin/bash

# Helper script to run qualibration tests in Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Qualibration Libs Test Runner${NC}"
echo "=================================="

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ docker-compose not found. Please install Docker Compose.${NC}"
    exit 1
fi

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [SCRIPT_NAME]"
    echo ""
    echo "Options:"
    echo "  --build     Build the Docker image before running"
    echo "  --up        Start the environment in background"
    echo "  --down      Stop the environment"
    echo "  --logs      Show logs from the container"
    echo "  --shell     Open a shell in the container"
    echo "  --help      Show this help message"
    echo ""
    echo "Available scripts:"
    echo "  plot_02a_resonator_spectroscopy_results.py"
    echo "  plot_02b_resonator_spectroscopy_vs_power_results.py"
    echo "  plot_02c_resonator_spectroscopy_vs_flux_results.py"
    echo "  plot_04b_power_rabi_results.py"
    echo ""
    echo "Examples:"
    echo "  $0 --build                                    # Build the environment"
    echo "  $0 --up                                       # Start environment in background"
    echo "  $0 plot_02a_resonator_spectroscopy_results.py # Run specific test"
    echo "  $0 --shell                                    # Open shell in container"
}

# Parse arguments
BUILD=false
UP=false
DOWN=false
LOGS=false
SHELL=false
SCRIPT_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD=true
            shift
            ;;
        --up)
            UP=true
            shift
            ;;
        --down)
            DOWN=true
            shift
            ;;
        --logs)
            LOGS=true
            shift
            ;;
        --shell)
            SHELL=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        -*)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
        *)
            SCRIPT_NAME="$1"
            shift
            ;;
    esac
done

# Handle different commands
if [ "$DOWN" = true ]; then
    echo -e "${YELLOW}🛑 Stopping qualibration-tests container...${NC}"
    docker-compose down
    exit 0
fi

if [ "$LOGS" = true ]; then
    echo -e "${BLUE}📋 Showing logs from qualibration-tests container...${NC}"
    docker-compose logs -f qualibration-tests
    exit 0
fi

if [ "$SHELL" = true ]; then
    echo -e "${BLUE}🐚 Opening shell in qualibration-tests container...${NC}"
    docker-compose exec qualibration-tests bash
    exit 0
fi

if [ "$UP" = true ]; then
    echo -e "${YELLOW}🚀 Starting qualibration-tests environment in background...${NC}"
    docker-compose up -d
    echo -e "${GREEN}✅ Environment started! Use '$0 --shell' to access the container.${NC}"
    exit 0
fi

# Build if requested
if [ "$BUILD" = true ]; then
    echo -e "${YELLOW}🔨 Building Docker image...${NC}"
    docker-compose build
fi

# If no script specified, start interactive environment
if [ -z "$SCRIPT_NAME" ]; then
    echo -e "${YELLOW}🚀 Starting qualibration-tests environment...${NC}"
    docker-compose up
    exit 0
fi

# Validate script name
SCRIPT_PATH="scripts/$SCRIPT_NAME"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}❌ Script not found: $SCRIPT_PATH${NC}"
    echo "Available scripts:"
    ls -1 scripts/*.py 2>/dev/null | sed 's/scripts\//  /' || echo "  No scripts found"
    exit 1
fi

# Run the specific script
echo -e "${YELLOW}🚀 Running $SCRIPT_NAME in Docker...${NC}"
echo -e "${BLUE}Command: python $SCRIPT_PATH --save${NC}"
echo ""

docker-compose run --rm qualibration-tests python "$SCRIPT_PATH" --save

echo ""
echo -e "${GREEN}✅ Test completed! Check scripts/plots/ for generated plots.${NC}"



