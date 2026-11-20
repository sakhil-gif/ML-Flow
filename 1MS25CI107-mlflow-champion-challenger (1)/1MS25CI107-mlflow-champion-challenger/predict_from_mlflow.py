"""Load the Champion through its MLflow alias and classify new sentences (no Hugging Face code here).
Usage: python predict_from_mlflow.py            -> uses @champion
       python predict_from_mlflow.py challenger -> uses @challenger
Student: S Akhil | USN: 1MS25CI107"""
import os, sys, mlflow, pandas as pd
from mlflow.tracking import MlflowClient

REGISTERED = "1MS25CI107-sentiment-model"
ALIAS = sys.argv[1] if len(sys.argv) > 1 else "champion"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

mv = MlflowClient().get_model_version_by_alias(REGISTERED, ALIAS)
uri = f"models:/{REGISTERED}@{ALIAS}"
print(f"Loading {uri}")
print(f"  -> version {mv.version} | hf_model_id = {mv.tags.get('hf_model_id')} | run = {mv.tags.get('run_name')}")
model = mlflow.pyfunc.load_model(uri)

new = pd.read_csv("data/new_data.csv")
out = model.predict(new[["text"]])
out.insert(0, "sample_id", new.sample_id)
out.insert(2, "expected", new.expected)
out["model_version"] = mv.version; out["hf_model_id"] = mv.tags.get("hf_model_id")
os.makedirs("outputs", exist_ok=True)
fname = f"outputs/new_data_predictions{'' if ALIAS == 'champion' else '_' + ALIAS}.csv"
out.to_csv(fname, index=False)
pd.set_option("display.max_colwidth", 60); pd.set_option("display.width", 160)
print(out[["sample_id", "text", "expected", "prediction", "confidence"]].to_string(index=False))
print(f"\nCorrect on new data: {(out.prediction == out.expected).sum()} / {len(out)}")
print(f"Saved {fname}")
