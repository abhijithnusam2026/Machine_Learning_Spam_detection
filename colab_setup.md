# Colab Setup Guide

To run the training for the DistilBERT Scam Classifier on Google Colab (to utilize the free GPUs), follow these steps:

1. **Upload the Notebook or Scripts**
   - Upload the `scripts/train_scam_classifier.py` and `scripts/predict_scam.py` directly to the Colab session, OR clone the entire repository into the Colab environment.

2. **Clone the Repository in Colab (Recommended)**
   Open a new Colab Notebook and run the following in a cell:
   ```bash
   !git clone https://github.com/your-username/intent-classif.git
   %cd intent-classif
   ```

3. **Install Dependencies**
   ```bash
   !pip install -q transformers datasets scikit-learn pandas torch kagglehub
   ```

4. **Set Up Kaggle API (To download the dataset in Colab)**
   ```python
   import os
   os.environ["KAGGLE_API_TOKEN"] = "KGAT_52b97680eef1fe65fb550c7d47f84c99"
   ```
   *Note: If you run the download script provided in the repository, it will automatically use this token.*
   ```bash
   !python data/raw/download.py
   ```

5. **Run the Training Script**
   Ensure the notebook's runtime type is set to **T4 GPU** (Runtime -> Change runtime type -> Hardware accelerator: GPU).
   
   ```bash
   !python scripts/train_scam_classifier.py --data data/raw/composite_train.csv --output_dir ./scam-classifier-model --epochs 4 --batch_size 16
   ```

6. **Test the Model**
   ```bash
   !python scripts/predict_scam.py --model_dir ./scam-classifier-model --text "URGENT: Your account has been compromised. Click here to reset your password immediately."
   ```

7. **Download the Trained Model**
   Once training is complete, you can zip and download the model folder from Colab:
   ```bash
   !zip -r scam-classifier-model.zip scam-classifier-model
   from google.colab import files
   files.download('scam-classifier-model.zip')
   ```
