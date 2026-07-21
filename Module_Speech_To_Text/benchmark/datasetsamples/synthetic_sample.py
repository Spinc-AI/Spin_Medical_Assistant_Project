"""
synthetic_sample.py — build 10 synthetic noisy clips from google/fleurs (fa_ir)
WITHOUT downloading the full dataset.

10 FINAL clips = 2 base FLEURS clips x 5 SNRs [20, 15, 10, 5, 0] dB, additive white
Gaussian noise (the notebook's Round-3 placeholder — swap add_noise_at_snr for
MUSAN/DEMAND/RIR later). Only 2 source clips are needed.

Minimum download: test.tsv (~0.71 MB) + leading ~1.5 MB of test.tar.gz (2 clips)
~= ~2 MB. The noise is generated in-memory — zero extra download.

Requires:  pip install requests soundfile numpy
Output:    samples/synthetic/clip_00_snr20.wav ... clip_09_snr0.wav  +  metadata.csv
"""
import os, io, csv, tarfile
import numpy as np
import soundfile as sf
import requests

REPO        = "google/fleurs"
LANG, SPLIT = "fa_ir", "test"
N_BASE      = 2
SNR_DB      = [20, 15, 10, 5, 0]
NOISE_TYPE  = "gaussian"
OUT_DIR     = os.path.join("samples", "synthetic")

BASE     = f"https://huggingface.co/datasets/{REPO}/resolve/main/data/{LANG}"
TSV_URL  = f"{BASE}/{SPLIT}.tsv"
TAR_URL  = f"{BASE}/audio/{SPLIT}.tar.gz"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
HEADERS  = {"Accept-Encoding": "identity"}
if HF_TOKEN:
    HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"


def add_noise_at_snr(clean, snr_db, rng):
    """Additive white Gaussian noise scaled to the requested SNR (notebook's fn)."""
    p_sig   = float(np.mean(clean.astype(np.float64) ** 2)) + 1e-12
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    noise   = rng.normal(0.0, np.sqrt(p_noise), size=clean.shape).astype(np.float32)
    return (clean + noise).astype(np.float32)


def load_transcripts():
    r = requests.get(TSV_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    refs = {}
    for line in r.text.splitlines():
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        refs[cols[1].strip()] = (cols[3] if len(cols) > 3 and cols[3].strip() else cols[2]).strip()
    return refs


def fetch_base_clips(n):
    """Stream the FLEURS test tar and return the first n clips (stops early)."""
    refs, clips = load_transcripts(), []
    with requests.get(TAR_URL, headers=HEADERS, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = False
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
            for member in tar:
                if not (member.isfile() and member.name.endswith(".wav")):
                    continue
                fname = os.path.basename(member.name)
                arr, sr = sf.read(io.BytesIO(tar.extractfile(member).read()), dtype="float32")
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
                clips.append({"file": fname, "array": arr.astype(np.float32),
                              "sr": sr, "reference": refs.get(fname, "")})
                if len(clips) >= n:
                    break
    return clips


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = fetch_base_clips(N_BASE)
    print(f"Fetched {len(base)} base FLEURS clips -> {len(base)*len(SNR_DB)} noisy clips")
    rng = np.random.default_rng(1234)

    rows, n = [], 0
    for b in base:
        for snr in SNR_DB:
            noisy = add_noise_at_snr(b["array"], snr, rng)
            out_wav = os.path.join(OUT_DIR, f"clip_{n:02d}_snr{snr}.wav")
            sf.write(out_wav, noisy, b["sr"])
            rows.append({
                "clip_id": n,
                "filename": os.path.basename(out_wav),
                "source_file": b["file"],
                "reference": b["reference"],
                "duration_sec": round(len(noisy) / b["sr"], 3),
                "sample_rate": b["sr"],
                "snr_db": snr,
                "noise_type": NOISE_TYPE,
            })
            print(f"  [{n}] {b['file']}  snr={snr}dB  {len(noisy)/b['sr']:.1f}s")
            n += 1

    with open(os.path.join(OUT_DIR, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {n} synthetic clips + metadata.csv -> {OUT_DIR}")


if __name__ == "__main__":
    main()
