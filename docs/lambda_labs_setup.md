# Lambda Labs Setup Guide

To run this project end-to-end on a powerful GPU machine (like those provided by Lambda Labs), follow these instructions to configure your instance.

## 1. Provision the Instance
1. Go to the [Lambda Labs Cloud Dashboard](https://cloud.lambdalabs.com/).
2. Launch a new instance.
   - For **DistilBERT only (College Project)**: A 1x RTX A4000 or A6000 instance is more than enough.
   - For **vLLM LLM Serving (Resume Project)**: Choose an instance with at least 1x A100 or 2x A6000 GPUs to ensure you have enough VRAM for the 3B/7B models.
3. Add your local SSH key during the setup.
4. Once the instance is running, copy the SSH command provided by the dashboard (e.g., `ssh ubuntu@192.168.x.x`).

## 2. SSH into the Instance
From your local terminal or VSCode (using the "Remote - SSH" extension):
```bash
ssh ubuntu@<YOUR_LAMBDA_IP>
```
*Note: If using VSCode, you can connect directly and open the remote folder, allowing you to edit files on the server directly from your local VSCode IDE.*

## 3. Clone the Repository
```bash
git clone -b feature/deliverable-03 https://github.com/tanu320/2026SU_MS_DSP_422-DL_SEC61_Machine_Learning_Spam_detection.git intent-classif
cd intent-classif
```

## 4. Run the Setup Script
We have provided an automated script that installs Python virtual environments, updates dependencies, installs Docker, and configures the NVIDIA Container Toolkit.

```bash
chmod +x scripts/setup_lambda_env.sh
./scripts/setup_lambda_env.sh
```

## 5. Configure Credentials
Copy the example environment file and add your tokens (like your W&B API Key, Hugging Face Token, and Kaggle API Token):
```bash
cp .env.example .env
nano .env 
```
*Make sure `KAGGLE_API_TOKEN` is set in the `.env` file to easily download the scam dataset!*

## 6. Run Training & Serving
Activate your environment:
```bash
source .venv/bin/activate
```

**To Train the Scam Classifier:**
```bash
python data/raw/download.py
python scripts/train_scam_classifier.py --data data/raw/composite_train.csv --output_dir ./scam-classifier-model --epochs 4 --batch_size 16
```

**To Start the Dual-Endpoint API:**
```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```
