# Use Python 3.11 as base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone the qua-libs repository
RUN git clone https://github.com/qua-platform/qua-libs.git .

# Install Python dependencies manually to avoid build issues
RUN pip install --no-cache-dir \
    quam>=0.4.0 \
    qualang-tools>=0.19.0 \
    qualibrate>=0.2.1 \
    qm-qua>=1.2.1 \
    xarray>=2024.7.0 \
    plotly>=5.24.1 \
    lmfit>=1.3.3 \
    scipy>=1.13.1 \
    qiskit-experiments>=0.9.0 \
    tqdm>=4.67.1 \
    pexpect>=4.8.0 \
    "qiskit<2.0" \
    "quam-builder@git+https://github.com/qua-platform/quam-builder.git"

# Note: Both qualibration-libs and superconducting calibrations will be installed at runtime via volume mount

# Copy the automated setup script
COPY auto_setup.py /app/qualibration-libs/auto_setup.py

# Copy preliminary datasets zip file if it exists (optional)
COPY preliminary_datasets.zip* /app/

# Make the setup script executable
RUN chmod +x /app/qualibration-libs/auto_setup.py

# Set the working directory back to qualibration-libs
WORKDIR /app/qualibration-libs

# Create directories for data and plots
RUN mkdir -p /app/qualibration_graphs/superconducting/data/QPU_project/2025-10-01
RUN mkdir -p /app/qualibration-libs/scripts/plots

# Set the default command
CMD ["bash"]
