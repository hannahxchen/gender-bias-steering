import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px

COLORS = {"MD": '#1F77B4', "WMD": '#FF7F0E', "before": "#4C78A8", "after": "#E45756"}

def load_json_file(filepath):
    return  json.load(open(filepath, "r"))

def load_projections(artifact_dir):
    return np.array([x["projection"] for x in load_json_file(os.path.join(artifact_dir, "validation/projections.json"))])

def load_val_data(baseline_dir):
    return pd.DataFrame.from_records(load_json_file(os.path.join(baseline_dir, "datasplits/val.json")))

def load_debiased_results(artifact_dir):
    debiased_results = load_json_file(os.path.join(artifact_dir, "validation/debiased_results.json"))
    debiased_scores = load_json_file(os.path.join(artifact_dir, "validation/debiased_scores.json"))
    for x in debiased_results:
        for y in debiased_scores:
            if y["layer"] == x["layer"]:
                x["bias_scores"] = y["bias_scores"]
                x["normalized_bias_scores"] = y["normalized_bias_scores"]
                break
    return debiased_results

def load_projection_correlation(artifact_dir):
    return load_json_file(os.path.join(artifact_dir, "validation/proj_correlation.json"))
