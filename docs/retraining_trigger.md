# Retraining Trigger and Maintenance Strategy

In a production setting, our ML models for summarization and intent classification can degrade over time due to **data drift** (e.g., shifts in customer demographics, new product launches, or seasonal issues). We must monitor this and trigger retraining pipelines proactively.

## Monitoring Signals

1. **Embedding Drift (Primary Unsupervised Trigger)**:
   - We run a scheduled job (e.g., daily or weekly) using `scripts/detect_embedding_drift.py`.
   - Incoming call transcripts are embedded and compared against the training baseline centroid.
   - **Threshold**: If the cosine distance exceeds the threshold (e.g., `> 0.12`) for **3 consecutive days**, an alert is raised to evaluate retraining.

2. **Downstream Application Metrics (Primary Supervised Trigger)**:
   - If downstream human-in-the-loop (HITL) auditing reveals intent classification accuracy drops below an acceptable SLA (e.g., `< 85%`).
   - If agents report high frequency of poor summarization or missing key entities.

3. **API Performance**:
   - If `call_center_api_errors_total` spikes specifically due to model inference bounds (e.g., generation going off the rails or hitting token limits frequently).

## The Retraining Workflow

When a trigger fires, the following steps are executed:

1. **Data Curation**: Pull the last N days of drifted/flagged data and route a sample to human labelers.
2. **Dataset Augmentation**: Combine the newly labeled data with the existing baseline training set to prevent catastrophic forgetting.
3. **Fine-Tuning**: Run the LoRA/QLoRA pipeline (`scripts/train_lora_qwen.py`) on the updated dataset.
4. **Evaluation**: Use `scripts/evaluate_phase1.py` to compare the candidate adapter against the current production adapter on a held-out gold evaluation set.
5. **Deployment**: If metrics improve, quantize the new adapter and perform a blue-green or canary rollout to the Kubernetes cluster using the FastAPI endpoints.

By automating the drift detection script and linking it to an alerting system (like PagerDuty or a Slack webhook), we ensure the system adapts dynamically to changing call center trends.
