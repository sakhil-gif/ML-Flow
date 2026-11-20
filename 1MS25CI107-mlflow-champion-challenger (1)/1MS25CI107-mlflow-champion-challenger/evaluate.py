"""Run Model A (Champion) and Model B (Challenger) on the same SST-2 sentences and log everything to MLflow.
Student: S Akhil | USN: 1MS25CI107"""
import json, os, time, gc
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow, torch, transformers, sklearn
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from hf_wrapper import HFSentimentModel, classify

USN = "1MS25CI107"
SEED, N_PER_CLASS, MAX_LENGTH, BATCH_SIZE = 107, 100, 128, 1
EXPERIMENT = f"{USN}-hf-sentiment-champion-challenger"
# Champion rule (written in README BEFORE the final run)
PRIMARY, MARGIN, LATENCY_LIMIT = "macro_f1", 2.0, 4.0
MODELS = {
    "A": {"short": "distilbert-sst2", "id": "distilbert-base-uncased-finetuned-sst-2-english", "role": "champion"},
    "B": {"short": "bert-base-sst2", "id": "textattack/bert-base-uncased-SST-2", "role": "challenger"},
}
METRICS = ["accuracy", "macro_f1", "precision_pos", "recall_pos", "recall_neg", "avg_latency_ms", "total_inference_s"]
LABELS = ["negative", "positive"]
os.makedirs("data", exist_ok=True); os.makedirs("outputs", exist_ok=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment(EXPERIMENT)
torch.manual_seed(SEED); torch.set_num_threads(max(1, os.cpu_count() or 1))

# ---- 1. shared evaluation set: 200 SST-2 validation sentences (100 negative + 100 positive, seed 107) ----
p = hf_hub_download("stanfordnlp/sst2", "data/validation-00000-of-00001.parquet", repo_type="dataset")
df = pd.read_parquet(p)
ev = pd.concat([df[df.label == 0].sample(N_PER_CLASS, random_state=SEED), df[df.label == 1].sample(N_PER_CLASS, random_state=SEED)]
               ).sample(frac=1, random_state=SEED).reset_index(drop=True)
ev = pd.DataFrame({"sample_id": [f"s_{i:03d}" for i in range(1, len(ev) + 1)], "sst2_idx": ev["idx"],
                   "text": ev.sentence.str.strip(), "reference": ev.label.map({0: "negative", 1: "positive"})})
ev.to_csv("data/eval_inputs.csv", index=False)
refs = ev.reference.tolist()
print(f"Evaluation set: {len(ev)} SST-2 validation sentences ({(ev.reference == 'negative').sum()} negative / "
      f"{(ev.reference == 'positive').sum()} positive) -> data/eval_inputs.csv")

tags = {"student_usn": USN, "dataset_name": "stanfordnlp/sst2 (validation)",
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "mlflow_version": mlflow.__version__}
results = {}

with mlflow.start_run(run_name="compare_sentiment") as parent:
    mlflow.set_tags(tags)
    for key, m in MODELS.items():
        run_name = f"{key}_sentiment_{m['short']}"
        print(f"\n=== {run_name}  ({m['id']}) ===")
        tok = AutoTokenizer.from_pretrained(m["id"])
        mdl = AutoModelForSequenceClassification.from_pretrained(m["id"]).eval()
        classify(tok, mdl, ["warm-up sentence"], MAX_LENGTH)            # first call is slow, keep it out of the timing
        preds, p_pos, lat = [], [], []
        for t in ev.text:
            t0 = time.perf_counter()
            l, pp = classify(tok, mdl, [t], MAX_LENGTH, BATCH_SIZE)
            lat.append(time.perf_counter() - t0)
            preds.append(l[0]); p_pos.append(pp[0])
        met = {"accuracy": 100 * accuracy_score(refs, preds),
               "macro_f1": 100 * f1_score(refs, preds, average="macro"),
               "precision_pos": 100 * precision_score(refs, preds, pos_label="positive"),
               "recall_pos": 100 * recall_score(refs, preds, pos_label="positive"),
               "recall_neg": 100 * recall_score(refs, preds, pos_label="negative"),
               "avg_latency_ms": 1000 * np.mean(lat), "total_inference_s": float(np.sum(lat))}
        cm = confusion_matrix(refs, preds, labels=LABELS)
        results[key] = {**met, "hf_model_id": m["id"], "run_name": run_name, "cm": cm}
        for k2, v in met.items(): print(f"  {k2:18s} {v:.4f}")

        out = ev[["sample_id", "text", "reference"]].copy()
        out["prediction"] = preds; out["positive_prob"] = np.round(p_pos, 4)
        out["confidence"] = np.round([pp if pr == "positive" else 1 - pp for pr, pp in zip(preds, p_pos)], 4)
        out["correct"] = (out.prediction == out.reference).astype(int); out["latency_ms"] = np.round(np.array(lat) * 1000, 1)
        pfile = f"outputs/predictions_{m['short']}.csv"; out.to_csv(pfile, index=False)

        # confusion matrix plot for this model
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm): ax.text(j, i, int(v), ha="center", va="center", fontsize=14,
                                                     color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], LABELS); ax.set_yticks([0, 1], LABELS); ax.set_xlabel("predicted"); ax.set_ylabel("reference")
        ax.set_title(f"{key}: {m['short']}"); fig.tight_layout()
        cmfile = f"outputs/confusion_matrix_{m['short']}.png"; fig.savefig(cmfile, dpi=120); plt.close(fig)

        with mlflow.start_run(run_name=run_name, nested=True) as r:
            mlflow.set_tags({**tags, "role": m["role"]})
            mlflow.log_params({"hf_model_id": m["id"], "task": "sentiment-classification", "role": m["role"],
                               "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE, "labels": "negative,positive",
                               "eval_samples": len(ev), "seed": SEED, "device": "cpu"})
            mlflow.log_metrics(met)
            mfile = f"outputs/metrics_summary_{m['short']}.json"
            json.dump({k2: round(v, 4) for k2, v in met.items()}, open(mfile, "w"), indent=2)
            for f in (pfile, mfile, cmfile, "data/eval_inputs.csv"): mlflow.log_artifact(f)
            mlflow.log_table(out.head(20), artifact_file="sample_predictions.json")
            mdir = f"hf_models/{m['short']}"; tok.save_pretrained(mdir); mdl.save_pretrained(mdir)
            example_in = pd.DataFrame({"text": ["The labs at Ramaiah were genuinely useful."]})
            example_out = pd.DataFrame({"text": example_in.text, "prediction": ["positive"],
                                        "positive_prob": [0.99], "confidence": [0.99]})
            info = mlflow.pyfunc.log_model(name="model", python_model=HFSentimentModel(), code_paths=["hf_wrapper.py"],
                                           artifacts={"hf_model_dir": mdir},
                                           model_config={"hf_model_id": m["id"], "max_length": MAX_LENGTH},
                                           signature=mlflow.models.infer_signature(example_in, example_out),
                                           pip_requirements=["mlflow", "torch", "transformers", "pandas"])
            results[key].update(run_id=r.info.run_id, model_uri=info.model_uri)
            print(f"  logged model: {info.model_uri}")
        del tok, mdl; gc.collect()

    # ---- 2. comparison + decision (rule fixed before running) ----
    A, B = results["A"], results["B"]
    delta = B[PRIMARY] - A[PRIMARY]; ratio = B["avg_latency_ms"] / A["avg_latency_ms"]
    if ratio > LATENCY_LIMIT: winner, why = "A", f"Challenger is {ratio:.2f}x slower (> {LATENCY_LIMIT}x guardrail)"
    elif delta >= MARGIN: winner, why = "B", f"Challenger beats Champion by {delta:+.2f} macro-F1 points (>= {MARGIN}) within the latency guardrail ({ratio:.2f}x)"
    elif abs(delta) < MARGIN: winner = "A" if A["avg_latency_ms"] <= B["avg_latency_ms"] else "B"; why = f"|delta macro-F1| {abs(delta):.2f} < {MARGIN}, faster model kept"
    else: winner, why = "A", f"Challenger is worse by {delta:+.2f} macro-F1 points"
    mlflow.log_params({"primary_metric": PRIMARY, "margin": MARGIN, "latency_guardrail_x": LATENCY_LIMIT})
    mlflow.log_metrics({"delta_macro_f1": delta, "delta_accuracy": B["accuracy"] - A["accuracy"], "latency_ratio_B_over_A": ratio})
    mlflow.set_tags({"winner": f"{winner}_{MODELS[winner]['short']}", "decision_reason": why})

    comp = pd.DataFrame({k: {m2: round(results[k][m2], 4) for m2 in METRICS} for k in "AB"})
    comp.columns = [f"A_{MODELS['A']['short']}", f"B_{MODELS['B']['short']}"]; comp["B_minus_A"] = (comp.iloc[:, 1] - comp.iloc[:, 0]).round(4)
    comp.to_csv("outputs/model_comparison.csv")
    json.dump({"decision": {"winner": winner, "reason": why, "delta_macro_f1": round(delta, 4), "latency_ratio": round(ratio, 3)},
               **{k: {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in results[k].items() if k2 != "cm"} for k in "AB"}},
              open("outputs/metrics_summary.json", "w"), indent=2)

    # plot 1: every metric, A vs B
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    comp.loc[["accuracy", "macro_f1", "precision_pos", "recall_pos", "recall_neg"]].iloc[:, :2].plot.bar(ax=ax[0], rot=0)
    ax[0].set_ylim(70, 100); ax[0].set_title("Quality metrics, % (higher is better)")
    comp.loc[["avg_latency_ms"]].iloc[:, :2].plot.bar(ax=ax[1], rot=0); ax[1].set_title("Avg latency ms/sentence (lower is better)")
    fig.tight_layout(); fig.savefig("outputs/metric_comparison.png", dpi=120); plt.close(fig)

    pa = pd.read_csv(f"outputs/predictions_{MODELS['A']['short']}.csv")
    pb = pd.read_csv(f"outputs/predictions_{MODELS['B']['short']}.csv")
    # plot 2: confidence distribution, split into correct and wrong predictions
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, (k, pdf) in zip(ax, (("A", pa), ("B", pb))):
        a.hist([pdf.confidence[pdf.correct == 1], pdf.confidence[pdf.correct == 0]], bins=np.linspace(0.5, 1, 11),
               label=["correct", "wrong"], stacked=True)
        a.set_title(f"{k}: {MODELS[k]['short']} confidence"); a.set_xlabel("confidence of predicted class"); a.legend()
    ax[0].set_ylabel("sentences"); fig.tight_layout(); fig.savefig("outputs/confidence_distribution.png", dpi=120); plt.close(fig)

    mask = pa.prediction != pb.prediction
    dis = pa[mask][["sample_id", "text", "reference"]].assign(pred_A=pa.prediction[mask], conf_A=pa.confidence[mask],
                                                               pred_B=pb.prediction[mask], conf_B=pb.confidence[mask])
    dis.to_csv("outputs/disagreements.csv", index=False)
    for f in ("outputs/model_comparison.csv", "outputs/metrics_summary.json", "outputs/metric_comparison.png",
              "outputs/confidence_distribution.png", "outputs/disagreements.csv", "outputs/qualitative_examples.csv", "README.md"):
        if os.path.exists(f): mlflow.log_artifact(f)
    json.dump({k: {"run_id": results[k]["run_id"], "hf_model_id": results[k]["hf_model_id"]} for k in "AB"} | {"winner": winner},
              open("outputs/model_runs.json", "w"), indent=2)

print("\n=== Champion-Challenger decision ===")
print(comp.to_string())
print(f"Winner: {winner} ({MODELS[winner]['id']}) -> {why}")
print(f"Sentences where A and B disagree: {len(dis)} of {len(ev)}")
