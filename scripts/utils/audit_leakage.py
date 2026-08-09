import pandas as pd
import numpy as np
from collections import Counter
import re

print("Loading data...")
train = pd.read_csv("data/train.csv")

scam = train[train['label'] == 1]
legit = train[train['label'] == 0]

print(f"\n--- 1. Length Bias Check ---")
scam_len = scam['text'].apply(lambda x: len(str(x).split())).mean()
legit_len = legit['text'].apply(lambda x: len(str(x).split())).mean()
print(f"Average Scam Length: {scam_len:.1f} words")
print(f"Average Legit Length: {legit_len:.1f} words")

print(f"\n--- 2. Vocabulary Leakage Check ---")
def get_words(df):
    words = []
    for text in df['text']:
        words.extend(re.findall(r'\b\w+\b', str(text).lower()))
    return Counter(words)

scam_words = get_words(scam)
legit_words = get_words(legit)

scam_top = [w for w, c in scam_words.most_common(20)]
legit_top = [w for w, c in legit_words.most_common(20)]

print("Top 10 words in Scams:", scam_top[:10])
print("Top 10 words in Legits:", legit_top[:10])

unique_scam_words = [w for w in scam_top if w not in legit_top]
print("Top words unique to Scams:", unique_scam_words[:5])

print(f"\n--- 3. Structural/Regex Leakage Check ---")
def check_regex(pattern, df):
    return df['text'].apply(lambda x: bool(re.search(pattern, str(x)))).mean()

print(f"Contains '[' (Kaggle brackets) -> Scam: {check_regex(r'\[', scam):.1%}, Legit: {check_regex(r'\[', legit):.1%}")
print(f"Contains 'suspect:' -> Scam: {check_regex(r'suspect:', scam):.1%}, Legit: {check_regex(r'suspect:', legit):.1%}")
print(f"Contains 'innocent:' -> Scam: {check_regex(r'innocent:', scam):.1%}, Legit: {check_regex(r'innocent:', legit):.1%}")
