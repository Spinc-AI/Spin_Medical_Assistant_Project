"""
fleurs_sample.py — draw 10 audio samples from google/fleurs (fa_ir, test split)
WITHOUT downloading the full dataset.

Minimum download: test.tsv (~0.71 MB) + the leading ~8 MB of test.tar.gz, streamed
sequentially and stopped after 10 clips  ~=  ~9 MB
(the full fa_ir test audio is 657 MB / 871 clips).

Why the tar and NOT the parquet: FLEURS' HF parquet export packs the entire fa_ir
test split into a SINGLE 844 MB row group, so parquet-streaming would pull all 844 MB.
The original tar.gz can be read sequentially and stopped early.

Requires:  pip install requests soundfile numpy
Output:    samples/fleurs_clean/clip_00.wav ... clip_09.wav  +  metadata.csv
"""
import os, io, csv, tarfile
import numpy as np
import soundfile as sf
import requests

REPO    = "google/fleurs"
LANG    = "fa_ir"
SPLIT   = "test"
N_CLIPS = 10
OUT_DIR = os.path.join("samples", "fleurs_clean")

BASE     = f"https://huggingface.co/datasets/{REPO}/resolve/main/data/{LANG}"
TSV_URL  = f"{BASE}/{SPLIT}.tsv"
TAR_URL  = f"{BASE}/audio/{SPLIT}.tar.gz"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
HEADERS  = {"Accept-Encoding": "identity"}          # keep byte ranges / gzip intact
if HF_TOKEN:
    HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"


def load_transcripts():
    """filename -> reference, from the header-less, tab-separated FLEURS TSV.
    Columns: 0=id 1=file_name 2=raw_transcription 3=transcription ..."""
    r = requests.get(TSV_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    refs = {}
    for line in r.text.splitlines():
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        ref = (cols[3] if len(cols) > 3 and cols[3].strip() else cols[2]).strip()
        refs[cols[1].strip()] = ref
    return refs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    refs = load_transcripts()
    print(f"Loaded {len(refs)} transcripts from {SPLIT}.tsv")

    rows, n = [], 0
    with requests.get(TAR_URL, headers=HEADERS, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = False             # we want the raw gzip stream
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
            for member in tar:                      # sequential -> stops early
                if not (member.isfile() and member.name.endswith(".wav")):
                    continue
                fname = os.path.basename(member.name)
                arr, sr = sf.read(io.BytesIO(tar.extractfile(member).read()), dtype="float32")
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
                out_wav = os.path.join(OUT_DIR, f"clip_{n:02d}.wav")
                sf.write(out_wav, arr, sr)
                rows.append({
                    "clip_id": n,
                    "filename": os.path.basename(out_wav),
                    "source_file": fname,
                    "reference": refs.get(fname, ""),
                    "duration_sec": round(len(arr) / sr, 3),
                    "sample_rate": sr,
                })
                print(f"  [{n}] {fname}  {len(arr)/sr:.1f}s  {refs.get(fname, '')[:50]}")
                n += 1
                if n >= N_CLIPS:
                    break

    with open(os.path.join(OUT_DIR, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {n} clips + metadata.csv -> {OUT_DIR}")


if __name__ == "__main__":
    main()
