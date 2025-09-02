#!/bin/bash

# create_environment.sh - Sets up development environment

ENV_NAME="ml-project"

echo "🚀 Setting up ML development environment..."

# Check if Anaconda/Miniconda is installed
if ! command -v conda &> /dev/null; then
    echo "📦 Anaconda/Miniconda not found. Installing Miniconda..."
    
    # Download and install Miniconda for Windows
    curl -o miniconda.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
    echo "Please run miniconda.exe manually and restart this script"
    exit 1
else
    echo "✅ Conda is already installed"
fi

# Remove existing environment if it exists
if conda env list | grep -q $ENV_NAME; then
    echo "🗑️ Removing existing $ENV_NAME environment..."
    conda env remove -n $ENV_NAME -y
fi

# Create new conda environment
echo "🔨 Creating new conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.9 -y

# Activate environment
echo "🔄 Activating environment: $ENV_NAME"
conda activate $ENV_NAME

# Install packages from dev_requirements.txt
if [ -f "dev_requirements.txt" ]; then
    echo "📚 Installing packages from dev_requirements.txt..."
    pip install -r dev_requirements.txt
else
    echo "❌ dev_requirements.txt not found!"
    exit 1
fi

# Install Jupyter kernel for this environment
echo "🔧 Setting up Jupyter kernel..."
python -m ipykernel install --user --name=$ENV_NAME --display-name="Python ($ENV_NAME)"

echo "✅ Environment setup complete!"
echo "🎯 To activate the environment, run: conda activate $ENV_NAME"
