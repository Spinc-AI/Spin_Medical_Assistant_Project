"""
psrb_sample.py — draw 10 audio samples from PartAI/PSRB WITHOUT downloading the
full dataset.

Minimum download: Labels.csv (~0.10 MB) + 10 individual wavs (avg ~1.17 MB each,
~12 MB total; less if you keep the shortest clips). Full dataset is 402 MB / 344 wavs.
PSRB is an 'audiofolder' repo (raw Labels.csv + Files/*.wav), so we fetch only the
files we need with hf_hub_download instead of snapshot_download (which pulls the
whole repo).

NOTE: Labels.csv's header is MISALIGNED with its data (the transcript actually sits
under the column labeled 'audio_duration'). Columns are detected by CONTENT, not by
header name — same trick the source notebook uses.

Requires:  pip install huggingface_hub pandas soundfile numpy
Output:    samples/psrb/clip_00.wav ... clip_09.wav  +  metadata.csv
"""
import os, re, csv
import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download

REPO       = "PartAI/PSRB"
N_CLIPS    = 10
DROP_CLEAN = True          # keep only non-clean (noisy/real) rows, like the notebook
OUT_DIR    = os.path.join("samples", "psrb")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    labels_path = hf_hub_download(REPO, "Labels.csv", repo_type="dataset")
    labels = pd.read_csv(labels_path)
    print(f"Labels.csv: {len(labels)} rows. Columns: {list(labels.columns)}")

    # detect columns by content (headers are misaligned)
    fa     = re.compile(r"[؀-ۿ]")                       # Arabic/Persian block
    sample = labels.head(50).astype(str)
    path_col = next((c for c in labels.columns
                     if sample[c].str.contains(r"\.wav", case=False).mean() > 0.5),
                    labels.columns[0])
    text_col = max((c for c in labels.columns if c != path_col),
                   key=lambda c: sample[c].map(lambda s: len(fa.findall(s))).mean())
    print(f"Detected  path_col='{path_col}'  text_col='{text_col}'  (header names ignored)")

    if DROP_CLEAN and "acoustic_environment" in labels.columns:
        labels = labels[labels["acoustic_environment"].astype(str).str.strip().str.lower() != "clean"]
        print(f"After dropping 'clean': {len(labels)} rows")

    rows, n = [], 0
    for _, row in labels.reset_index(drop=True).iterrows():
        rel   = str(row[path_col]).strip()
        fname = rel if rel.startswith("Files/") else f"Files/{os.path.basename(rel)}"
        try:
            wav_path = hf_hub_download(REPO, fname, repo_type="dataset")
        except Exception as e:
            print(f"  skip {fname}: {e}")
            continue
        arr, sr = sf.read(wav_path, dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        out_wav = os.path.join(OUT_DIR, f"clip_{n:02d}.wav")
        sf.write(out_wav, arr, sr)
        rows.append({
            "clip_id": n,
            "filename": os.path.basename(out_wav),
            "source_file": os.path.basename(rel),
            "reference": str(row[text_col]).strip(),
            "duration_sec": round(len(arr) / sr, 3),
            "sample_rate": sr,
            "acoustic_environment": row.get("acoustic_environment"),
            "spontaneous": row.get("spontaneous"),
        })
        print(f"  [{n}] {os.path.basename(rel)}  {len(arr)/sr:.1f}s  {str(row[text_col]).strip()[:50]}")
        n += 1
        if n >= N_CLIPS:
            break

    with open(os.path.join(OUT_DIR, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {n} clips + metadata.csv -> {OUT_DIR}")


if __name__ == "__main__":
    main()
