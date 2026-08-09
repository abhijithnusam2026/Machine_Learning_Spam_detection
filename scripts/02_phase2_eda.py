"""
Generates basic dataset characteristics (Size, Class Balance, Length Distribution)
relevant for model training constraints (e.g. BERT max_seq_length).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_basic_eda(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # Calculate word counts (proxy for token length)
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    
    scams = df[df['label'] == 1]
    legits = df[df['label'] == 0]
    
    # 1. Plot Class Balance
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='label', hue='label', palette='viridis', legend=False)
    plt.title("Dataset Class Balance")
    plt.xlabel("Class (0: Legit, 1: Scam)")
    plt.ylabel("Count")
    balance_path = os.path.join(output_dir, "class_balance.png")
    plt.savefig(balance_path, bbox_inches='tight')
    plt.close()
    
    # 2. Plot Word Count Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='word_count', hue='label', bins=50, kde=True, palette='viridis')
    plt.title("Word Count Distribution (Scam vs Legit)")
    plt.xlabel("Word Count (Proxy for Token Length)")
    plt.ylabel("Frequency")
    plt.axvline(x=512, color='red', linestyle='--', label='BERT Max Tokens (512)')
    plt.legend()
    dist_path = os.path.join(output_dir, "word_count_dist.png")
    plt.savefig(dist_path, bbox_inches='tight')
    plt.close()
    
    # Generate Markdown Report
    abs_balance_path = os.path.abspath(balance_path)
    abs_dist_path = os.path.abspath(dist_path)
    
    markdown_output = f"""# Phase 2 Dataset Characteristics

## Overview
Basic statistics tracking the size and structural characteristics of the Phase 2 dataset prior to model training.

- **Total Transcripts:** {len(df)}
- **Scam Transcripts (Label 1):** {len(scams)} 
- **Legit Transcripts (Label 0):** {len(legits)} 

## Length Constraints for Model Training
This metric is critical for configuring the `max_seq_length` hyperparameter during ModernBERT fine-tuning.

| Metric | Overall | Scams (Label 1) | Legits (Label 0) |
|--------|---------|-----------------|------------------|
| **Mean Word Count** | {df['word_count'].mean():.1f} | {scams['word_count'].mean():.1f} | {legits['word_count'].mean():.1f} |
| **Max Word Count** | {df['word_count'].max()} | {scams['word_count'].max()} | {legits['word_count'].max()} |

## Visualizations

### Class Balance
![Class Balance]({abs_balance_path})

### Word Count Distribution (Truncation Risk)
![Word Count Distribution]({abs_dist_path})
*Note: The red dashed line represents the 512 token limit for standard BERT models.*
"""
    
    artifact_path = os.path.join(output_dir, "Phase2_Dataset_Summary.md")
    with open(artifact_path, "w") as f:
        f.write(markdown_output)
        
    print(f"Successfully generated Basic EDA and saved to {artifact_path}")

if __name__ == "__main__":
    generate_basic_eda("data/phase2_asr/raw_asr_transcripts.csv", "data/phase2_asr/plots")
