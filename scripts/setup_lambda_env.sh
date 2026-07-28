#!/bin/bash
# setup_lambda_env.sh
# Automates the setup of a Lambda Labs GPU instance (Ubuntu) for the Call Center Intelligence System

set -e

echo "=========================================="
echo "Starting Lambda Labs Instance Setup..."
echo "=========================================="

# 1. Update system packages
sudo apt-get update -y
sudo apt-get upgrade -y

# 2. Install essential tools (git, curl, wget, python3-venv)
sudo apt-get install -y git curl wget python3-venv python3-pip htop tmux

# 3. Setup Python Virtual Environment
echo "Setting up Python Virtual Environment..."
cd ~
if [ ! -d "intent-classif" ]; then
    echo "Please clone the repository first, e.g.: git clone -b feature/deliverable-03 https://github.com/tanu320/2026SU_MS_DSP_422-DL_SEC61_Machine_Learning_Spam_detection.git intent-classif"
    exit 1
fi

cd intent-classif
python3 -m venv .venv
source .venv/bin/activate

# 4. Install Project Dependencies
echo "Installing Python Dependencies..."
pip install --upgrade pip
pip install -e ".[dev,gpu]"
# specifically ensure colab requirements are present just in case
pip install transformers datasets scikit-learn pandas torch kagglehub

# 5. Setup Docker (for vLLM container)
echo "Installing Docker..."
if ! command -v docker &> /dev/null
then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "Docker installed successfully."
else
    echo "Docker is already installed."
fi

# 6. Install NVIDIA Container Toolkit (allows Docker to use GPU)
echo "Installing NVIDIA Container Toolkit..."
if ! command -v nvidia-ctk &> /dev/null
then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
else
    echo "NVIDIA Container Toolkit is already installed."
fi

echo "=========================================="
echo "Setup Complete!"
echo "Please re-login or run 'newgrp docker' to apply docker permissions."
echo "You can now run 'source .venv/bin/activate' to use the environment."
echo "=========================================="
