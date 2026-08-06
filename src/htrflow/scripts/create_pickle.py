import pandas as pd
import numpy as np

def cer(pred: str, gt: str) -> float:
    """Compute Character Error Rate (edit distance / length of GT)."""
    if not isinstance(pred, str) or not isinstance(gt, str) or not gt:
        return np.nan
    # Use Python stdlib difflib for edit distance if rapidfuzz/Levenshtein not available
    import difflib
    sm = difflib.SequenceMatcher(None, pred, gt)
    # convert similarity ratio to edit distance
    edit_ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            edit_ops.append(max(i2 - i1, j2 - j1))
        elif tag == "delete":
            edit_ops.append(i2 - i1)
        elif tag == "insert":
            edit_ops.append(j2 - j1)
    dist = sum(edit_ops)
    return dist / len(gt)

# Load your combined CSV
csv_path = "/home/amlpai05/erik/hackathon_2025/htrflow/outputs/cer_pred/cer_pred.csv"
df = pd.read_csv(csv_path)

# Compute CER row by row
df["cer"] = [
    cer(pred, gt) if isinstance(gt, str) and gt else np.nan
    for pred, gt in zip(df["htr_text"], df["best_gt_text"])
]

# Save updated DataFrame back to CSV or pickle
df.to_csv("/home/amlpai05/erik/hackathon_2025/htrflow/outputs/cer_pred_with_cer.csv", index=False)
df.to_feather("/home/amlpai05/erik/hackathon_2025/htrflow/outputs/pickle/htr_out_combined_with_cer.feather")

print("Added CER column. Example rows:")
print(df[["htr_text", "best_gt_text", "cer"]].head())