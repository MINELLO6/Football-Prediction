#!/bin/bash
# Script to create and set up the combined R and Python environment

# Create the conda environment from the YAML file
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate the environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate DataScienceEnv

# Install additional R packages that might not be properly installed through conda
echo "Installing additional R packages..."
R -e "if(!require('DiscreteWeibull')) install.packages('DiscreteWeibull', repos='https://mirrors.tuna.tsinghua.edu.cn/CRAN/')"
R -e "if(!require('penaltyblog')) install.packages('penaltyblog', repos='https://mirrors.tuna.tsinghua.edu.cn/CRAN/')"

# Install Bioconductor packages
echo "Installing Bioconductor packages..."
R -e "if(!require('BiocManager')) install.packages('BiocManager', repos='https://mirrors.tuna.tsinghua.edu.cn/CRAN/'); BiocManager::install(version='3.18')"
R -e "BiocManager::install(c('DESeq2', 'limma', 'edgeR', 'Seurat'), ask=FALSE)"

# Install GitHub R packages
echo "Installing GitHub R packages..."
R -e "if(!require('devtools')) install.packages('devtools', repos='https://mirrors.tuna.tsinghua.edu.cn/CRAN/'); devtools::install_github('immunogenomics/harmony')"

# Install any additional Python packages
echo "Installing additional Python packages..."
pip install --upgrade "jax[cpu]" # Example of a package that's better installed through pip

echo "Environment setup complete! Activate it with: conda activate DataScienceEnv"

# Optional: Display installed packages for verification
echo "Verifying R packages..."
R -e "installed.packages()[,c('Package', 'Version')]"

echo "Verifying Python packages..."
pip list