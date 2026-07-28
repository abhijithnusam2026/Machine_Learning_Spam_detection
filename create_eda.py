import nbformat as nbf

nb = nbf.v4.new_notebook()

md_intro = """# Deliverable 03: Interim Project Report
## Scam Alert System for Messages and Call Transcripts

This notebook covers:
1. Exploratory Data Analysis (EDA) of the composite dataset.
2. Baseline model evaluation.
"""

code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import warnings
warnings.filterwarnings('ignore')
"""

md_load = """## 1. Exploratory Data Analysis (EDA)"""

code_load = """# Load dataset
train_df = pd.read_csv('../data/raw/composite_train.csv')
test_df = pd.read_csv('../data/raw/composite_test.csv')

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")
train_df.head()
"""

code_eda = """# Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=train_df, x='label')
plt.title('Class Distribution (0 = Legit, 1 = Scam)')
plt.show()

# Text Lengths
train_df['text_length'] = train_df['text'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(8,5))
sns.histplot(data=train_df, x='text_length', hue='label', bins=50, kde=True)
plt.title('Text Length Distribution by Class')
plt.show()
"""

md_baseline = """## 2. Baseline Model (DistilBERT)
As proposed, we will use a fine-tuned DistilBERT model as our baseline. The script `scripts/train_scam_classifier.py` was used to train this model.
We will now evaluate its performance on our test set."""

code_baseline = """import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the trained model and tokenizer
# Assuming the model is saved in `scam-classifier-model`
# If you haven't run the training script yet, do so first: 
# !python ../scripts/train_scam_classifier.py --data ../data/raw/composite_train.csv
model_dir = '../scam-classifier-model'
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure you have trained the model using scripts/train_scam_classifier.py")
"""

code_train_baseline = """# Predict on test set (Sample of 100 for quick evaluation in notebook)
sample_test = test_df.sample(100, random_state=42)
y_test = sample_test['label'].tolist()
y_pred = []
y_prob = []

if 'model' in locals():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for text in sample_test['text']:
        inputs = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            
        pred_label = 1 if probs[1] > probs[0] else 0
        y_pred.append(pred_label)
        y_prob.append(probs[1].item())
"""

md_analysis = """## 3. Analysis and Interpretation"""

code_analysis = """# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Scam'], yticklabels=['Legit', 'Scam'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(md_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(md_load),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_code_cell(code_eda),
    nbf.v4.new_markdown_cell(md_baseline),
    nbf.v4.new_code_cell(code_baseline),
    nbf.v4.new_code_cell(code_train_baseline),
    nbf.v4.new_markdown_cell(md_analysis),
    nbf.v4.new_code_cell(code_analysis)
]

with open('notebooks/03_interim_project_report.ipynb', 'w') as f:
    nbf.write(nb, f)
