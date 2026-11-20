"""Sentiment helper + custom MLflow pyfunc wrapper (Lab 9 style).
Both models are binary (NEGATIVE / POSITIVE), but they name their labels differently,
so every label is normalised to "negative" / "positive" before scoring.
Student: S Akhil | USN: 1MS25CI107"""
import mlflow.pyfunc
import pandas as pd


def normalise_label(raw):
    """Map whatever the model's config calls a class to 'negative' or 'positive'."""
    r = str(raw).strip().lower()
    if r in ("negative", "neg", "label_0", "0"):
        return "negative"
    if r in ("positive", "pos", "label_1", "1"):
        return "positive"
    raise ValueError(f"Unknown sentiment label: {raw}")


def classify(tok, mdl, texts, max_length=128, batch_size=1):
    """Return (labels, positive_probabilities) for a list of sentences."""
    import torch
    id2label = {int(k): normalise_label(v) for k, v in mdl.config.id2label.items()}
    pos_idx = [i for i, l in id2label.items() if l == "positive"][0]
    labels, p_pos = [], []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        with torch.no_grad():
            probs = mdl(**enc).logits.softmax(-1)
        for row in probs:
            labels.append(id2label[int(row.argmax())])
            p_pos.append(float(row[pos_idx]))
    return labels, p_pos


class HFSentimentModel(mlflow.pyfunc.PythonModel):
    """Input: DataFrame with a `text` column. Output: DataFrame with text, prediction, positive probability, confidence."""

    def load_context(self, context):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        cfg = context.model_config or {}
        self.max_length = int(cfg.get("max_length", 128))
        path = context.artifacts["hf_model_dir"]
        self.tok = AutoTokenizer.from_pretrained(path)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].astype(str).tolist()
        labels, p_pos = classify(self.tok, self.mdl, texts, self.max_length)
        conf = [p if l == "positive" else 1 - p for l, p in zip(labels, p_pos)]
        return pd.DataFrame({"text": texts, "prediction": labels,
                             "positive_prob": [round(p, 4) for p in p_pos],
                             "confidence": [round(c, 4) for c in conf]})
