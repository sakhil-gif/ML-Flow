# 1MS25CI107 – Champion–Challenger Model Comparison with Hugging Face and MLflow
**Student:** S Akhil · **USN:** 1MS25CI107 · **Section:** B · Skill Lab-1 (CISL-1)

## 1. Task and models
| Item | Value |
|---|---|
| Task | Binary sentiment classification (negative / positive) |
| Dataset | SST-2 validation split (`stanfordnlp/sst2`, Hugging Face Hub) |
| Subset | 200 sentences: 100 negative and 100 positive, seed 107 → `data/eval_inputs.csv` |
| Model A – current Champion | `distilbert-base-uncased-finetuned-sst-2-english` (~67M, table row 1) |
| Model B – Challenger | `textattack/bert-base-uncased-SST-2` (~110M, table row 3) |
| Inference settings | CPU, max_length 128, batch_size 1, argmax over the two classes |

Pairing "1 vs 3" is one of the suggested classification pairings in the brief. Model B outputs `LABEL_0` / `LABEL_1`, so `hf_wrapper.py` (`normalise_label`) maps every model's labels to `negative` / `positive` before scoring.

## 2. How to run
```bash
pip install -r requirements.txt
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
# in a second terminal:
python evaluate.py                       # runs A and B on the same 200 sentences, logs everything, makes the decision
python register_champion.py              # registers both versions, sets @champion / @challenger aliases
python predict_from_mlflow.py            # loads models:/1MS25CI107-sentiment-model@champion and classifies data/new_data.csv
python predict_from_mlflow.py challenger # stretch goal: same script, other alias, no code change
```
MLflow experiment: `1MS25CI107-hf-sentiment-champion-challenger` · Registered model: `1MS25CI107-sentiment-model`
(To use another tracking server, set `MLFLOW_TRACKING_URI`. No run IDs are hard-coded.)

## 3. Champion rule (written before the final run)
- **Primary metric:** macro-F1 (%) on the shared 200-sentence set. **Higher is better.**
- **Margin:** B must beat A by **at least 2.0 macro-F1 points**. With 200 sentences one sentence is worth 0.5 points, so a gain under 2 points could come from 3–4 lucky sentences.
- **Guardrail:** B is rejected if its average latency is **more than 4×** A's. **Lower latency is better.**
- **Tie-breaker:** if |Δ macro-F1| < 2.0, keep the **faster** model.
- Secondary metrics (reported, not used for the decision): accuracy ↑, precision (positive) ↑, recall (positive) ↑, recall (negative) ↑, total inference time ↓.

## 4. Results
| Metric | A: DistilBERT-SST2 | B: BERT-base-SST2 | B − A |
|---|---:|---:|---:|
| Accuracy ↑ | 90.00 | 90.50 | +0.50 |
| **Macro-F1 ↑ (primary)** | **89.99** | **90.50** | **+0.51** |
| Precision (positive) ↑ | 87.74 | 89.32 | +1.58 |
| Recall (positive) ↑ | 93.00 | 92.00 | −1.00 |
| Recall (negative) ↑ | 87.00 | 89.00 | +2.00 |
| Avg latency (ms/sentence) ↓ | 47.64 | 88.20 | 1.85× |
| Total inference (s) ↓ | 9.53 | 17.64 | +8.11 |

A and B disagree on 13 of the 200 sentences (`outputs/disagreements.csv`): B is right on 7 of them and A on 6. Five are commented in `outputs/qualitative_examples.csv`.
Plots: `outputs/metric_comparison.png`, `outputs/confidence_distribution.png`, `outputs/confusion_matrix_*.png`.

## 5. Decision – Champion stays: Model A (`distilbert-base-uncased-finetuned-sst-2-english`)
The Challenger is slightly better, but not by enough for the rule to accept it. Its macro-F1 is 90.50 against 89.99, a gain of +0.51 points, well below the 2.0-point margin fixed before the run. BERT-base's latency is 1.85× DistilBERT's, which is inside the 4× guardrail, so the guardrail is not what decides it. The tie-breaker is: with |Δ macro-F1| < 2.0, the faster model is kept, and DistilBERT classifies a sentence in about 48 ms against 88 ms. Model A therefore keeps the `@champion` alias (version 1) and BERT-base is registered as version 2 with `@challenger`.

Is the difference meaningful? No. The two models disagree on only 13 sentences, and the whole +0.51 comes from B winning 7 of those against A's 6, i.e. one sentence. On a different random sample of 200 that could easily reverse. B's real edge is on negative sentences (recall 89% vs 87%), while A finds slightly more positives (93% vs 92%).

Trade-offs: BERT-base is about 1.6× bigger (110M vs 67M parameters) and almost 2× slower on CPU for practically the same accuracy. Both models struggle with sarcasm and mixed reviews (see the qualitative examples). On the 8 new sentences the Challenger got 8/8 and the Champion 6/8, both misses being sarcasm or a "confusing at first, but" sentence. That hints B may handle such cases better, but 8 hand-written sentences are far too few to change the decision, and the rule was fixed on the evaluation set.

## 6. Limitations and declaration
- Only 200 SST-2 sentences (movie-review snippets), so ±2 macro-F1 points is within noise. Results may differ on other domains such as tweets or product reviews.
- Both models were fine-tuned on SST-2 training data, so this measures in-domain performance.
- `siebert/sentiment-roberta-large-english` (row 2) was tried first as the Challenger, but loading and logging a 355M model ran out of memory on my 8 GB machine, so the suggested "1 vs 3" pairing was used instead. The rule was written before the final run and was not changed after seeing results.
- Latency was measured on a CPU and varies between runs (a warm-up call is excluded). The ratio is more reliable than the absolute ms.
- `predict_from_mlflow.py` never calls Hugging Face directly. It loads the model through the alias URI, and the pyfunc wrapper calls transformers inside MLflow. The version and HF model ID are read from the registry tags.
- New data (`data/new_data.csv`, 8 sentences with my own expected labels, including sarcasm and negation) was written by me and is not from SST-2.
- AI assistant used: Hark (AI assistant) helped write and debug the scripts and README. I can explain every line.

## 7. Repository structure
```
data/        eval_inputs.csv, new_data.csv
outputs/     predictions_*.csv, metrics_summary*.json, model_comparison.csv, disagreements.csv, qualitative_examples.csv,
             metric_comparison.png, confidence_distribution.png, confusion_matrix_*.png,
             new_data_predictions*.csv, model_runs.json, *_terminal.txt
screenshots/ MLflow runs, compare view, summary run, registry aliases, predict_from_mlflow.py and register terminal output
evaluate.py, hf_wrapper.py, register_champion.py, predict_from_mlflow.py, requirements.txt, .gitignore
```
