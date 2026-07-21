"""
common_voice_sample.py — draw 10 audio samples from the public Common Voice mirror
hezarai/common-voice-13-fa (test split) WITHOUT downloading the full dataset.

Minimum download: parquet footer (~0.17 MB) + first row group (100 rows, ~3.92 MB)
~= ~4 MB  (the full test split is 383 MB / 10,440 clips). HF streaming reads one
parquet row group at a time, and this mirror uses ~100-row groups, so .take(10)
only pulls the first group.

Audio bytes are mp3; we decode them with soundfile exactly like the source notebook
(decode=False + sf.read), which needs libsndfile >= 1.1 (already present in the
benchmark env). No torchcodec dependency.

Requires:  pip install "datasets>=2.18" soundfile numpy
Output:    samples/common_voice_fa/clip_00.wav ... clip_09.wav  +  metadata.csv
"""
import os, io, csv
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio

REPO    = "hezarai/common-voice-13-fa"
CONFIG  = "default"
SPLIT   = "test"
N_CLIPS = 10
OUT_DIR = os.path.join("samples", "common_voice_fa")


def decode_audio(val):
    """(mono float32, sr) from an HF audio value (dict with array/bytes/path)."""
    if isinstance(val, dict):
        if val.get("array") is not None:
            arr = np.asarray(val["array"], dtype=np.float32)
            sr  = int(val["sampling_rate"])
        elif val.get("bytes"):
            arr, sr = sf.read(io.BytesIO(val["bytes"]), dtype="float32")
        elif val.get("path"):
            arr, sr = sf.read(val["path"], dtype="float32")
        else:
            raise ValueError("audio dict has no array/bytes/path")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return arr.astype(np.float32), int(sr)
    raise ValueError(f"Unrecognized audio value: {type(val)}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Streaming {REPO} ({CONFIG}/{SPLIT}) ...")
    ds = load_dataset(REPO, CONFIG, split=SPLIT, streaming=True)

    feats     = ds.features or {}
    audio_col = next((c for c, f in feats.items() if isinstance(f, Audio)), "audio")
    ds        = ds.cast_column(audio_col, Audio(decode=False))   # keep raw mp3 bytes

    rows, n, ref_col = [], 0, None
    for row in ds:
        if ref_col is None:
            ref_col = next((c for c in ("sentence", "transcription", "text") if c in row), "")
        arr, sr = decode_audio(row[audio_col])
        ref = (row.get(ref_col) or "").strip() if ref_col else ""
        out_wav = os.path.join(OUT_DIR, f"clip_{n:02d}.wav")
        sf.write(out_wav, arr, sr)
        rows.append({
            "clip_id": n,
            "filename": os.path.basename(out_wav),
            "reference": ref,
            "duration_sec": round(len(arr) / sr, 3),
            "sample_rate": sr,
        })
        print(f"  [{n}] {len(arr)/sr:.1f}s  {ref[:50]}")
        n += 1
        if n >= N_CLIPS:
            break

    with open(os.path.join(OUT_DIR, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {n} clips + metadata.csv -> {OUT_DIR}")


if __name__ == "__main__":
    main()
