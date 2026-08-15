# Data Partition Flow

This diagram documents which source data is used by each pipeline step and which splits are protected from leakage.

```mermaid
flowchart TD
    A["Raw source datasets<br/>Kaggle SMS/Email/Phishing<br/>Teeconnie non-scam<br/>Synthetic LLM JSON<br/>Raw ASR transcripts"] --> B["02_build_datasets.py<br/>clean text, normalize labels,<br/>deduplicate by cleaned text"]

    B --> C["Canonical deduplicated dataset<br/>text, label, source_domain, source_dataset"]

    C --> D["global_test.csv<br/>20% frozen holdout<br/>stratified by label + source_domain"]
    C --> E["Train/validation pool<br/>remaining 80%"]

    E --> F["global_val.csv<br/>20% of remaining pool<br/>training-time validation only"]
    E --> G["global_train.csv<br/>model training only"]

    G --> H["ptq_calibration.csv<br/>256-row stratified sample<br/>from global_train only"]

    G --> I["03 DistilBERT baseline<br/>written_text only<br/>excluding synthetic"]
    F --> I

    G --> J["04 ModernBERT universal<br/>written_text only<br/>including synthetic"]
    F --> J

    G --> K["05 ModernBERT ASR retraining<br/>spoken_asr only"]
    F --> K

    H --> L["06 ONNX Runtime static INT8 PTQ<br/>calibration only"]
    K --> L
    K --> M["06 GGUF weight-only PTQ<br/>F16 / Q8_0 / Q4_K_M"]

    D --> N["Post-quantization classifier evaluation<br/>ONNX FP32 vs ONNX INT8<br/>GGUF F16 vs Q8/Q4"]
    L --> N
    M --> N

    O["data/large_audio_test/manifest.csv<br/>separate audio benchmark holdout"] --> P["07 Whisper quant benchmark<br/>F16 / BF16 / Q8_0 / Q4_K"]
    O --> Q["08 E2E combo benchmark<br/>Whisper variant + classifier variant"]
    N --> Q
    P --> Q

    Q --> R["09 best_pipeline_selection.py<br/>rank by accuracy, latency, size<br/>write serving config"]
```

## Split Contract

- `global_train.csv`: used for model fitting. It is the only source for `ptq_calibration.csv`.
- `global_val.csv`: used for training-time validation and checkpoint/model selection.
- `global_test.csv`: frozen final classifier holdout, used for post-training and post-quantization evaluation.
- `ptq_calibration.csv`: train-only calibration split for ONNX Runtime static INT8 PTQ. It is asserted to be disjoint from validation and test.
- `data/large_audio_test/manifest.csv`: separate audio benchmark holdout for ASR and full pipeline latency/quality tests.
