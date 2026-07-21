"""
synthetic_realistic_sample.py — build 10 clinic-REALISTIC noisy clips from
google/fleurs (fa_ir) WITHOUT downloading the full dataset.

10 FINAL clips = 2 base FLEURS clips x 5 SNRs [20, 15, 10, 5, 0] dB.
Profile (matches benchmark_noisy_kaggle.ipynb -> REALISTIC_PIPELINE):
    room reverb -> Gaussian (cheap-mic self-noise) -> band-limit -> mild clip
    -> Opus codec round-trip.
The SNR sweep shifts the additive-noise floor by (S - PROFILE_REF_SNR); since the
Gaussian base == PROFILE_REF_SNR(=10), its SNR equals the sweep value S directly.
Reverb and the mic chain are channel effects, applied at fixed strength.

Minimum download: test.tsv (~0.71 MB) + leading ~1.5 MB of test.tar.gz (2 clips)
~= ~2 MB. The degradation is generated in-memory — zero extra download.

Requires:  pip install requests soundfile numpy scipy
           optional: pyroomacoustics (realistic reverb; falls back to synthetic IR)
           optional: ffmpeg on PATH (codec round-trip; skipped if absent)
Output:    samples/synthetic_realistic/clip_00_snr20.wav ... clip_09_snr0.wav  +  metadata.csv
"""
import os, io, csv, tarfile, subprocess, tempfile
import numpy as np
import soundfile as sf
import requests
from scipy.signal import butter, sosfilt

REPO         = "google/fleurs"
LANG, SPLIT  = "fa_ir", "test"
N_BASE       = 2
SNR_DB       = [20, 15, 10, 5, 0]
PROFILE_REF_SNR = 10          # sweep midpoint: at this SNR the pipeline == the tuned mix
PROFILE_NAME = "clinic_realistic"
OUT_DIR      = os.path.join("samples", "synthetic_realistic")

BASE     = f"https://huggingface.co/datasets/{REPO}/resolve/main/data/{LANG}"
TSV_URL  = f"{BASE}/{SPLIT}.tsv"
TAR_URL  = f"{BASE}/audio/{SPLIT}.tar.gz"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
HEADERS  = {"Accept-Encoding": "identity"}
if HF_TOKEN:
    HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"


# ── level helpers ────────────────────────────────────────────────
def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-12)

def scale_to_snr(speech, noise, snr_db):
    target = _rms(speech) / (10.0 ** (snr_db / 20.0))
    return (noise * (target / _rms(noise))).astype(np.float32)

def _fit(x, n):
    if len(x) == 0:
        return np.zeros(n, np.float32)
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))))
    return x[:n].astype(np.float32)


# ── effect functions: fn(audio, sr, rng, **params) -> audio ──────
def reverb_synth(audio, sr, rng, rt60=0.4, **k):
    n = max(1, int(sr * rt60)); t = np.arange(n)
    ir = (rng.standard_normal(n) * np.exp(-6.908 * t / (rt60 * sr))).astype(np.float32)
    ir[0] += 1.0
    out = np.convolve(audio, ir)[:len(audio)]
    return (out * (_rms(audio) / _rms(out))).astype(np.float32)

def reverb_pyroom(audio, sr, rng, rt60=0.4, room=(4.0, 5.0, 3.0), **k):
    try:
        import pyroomacoustics as pra
        e_abs, max_order = pra.inverse_sabine(rt60, list(room))
        r = pra.ShoeBox(list(room), fs=sr, materials=pra.Material(e_abs), max_order=int(max_order))
        r.add_source([room[0] * 0.5, room[1] * 0.35, 1.2], signal=audio.astype(np.float64))
        r.add_microphone(np.array([room[0] * 0.5, room[1] * 0.65, 1.2]).reshape(3, 1))
        r.simulate()
        out = _fit(np.asarray(r.mic_array.signals[0], np.float32)[:len(audio)], len(audio))
        return (out * (_rms(audio) / _rms(out))).astype(np.float32)
    except Exception as e:
        print(f"    [reverb_pyroom -> synth fallback: {e}]")
        return reverb_synth(audio, sr, rng, rt60=rt60)

def add_gaussian(audio, sr, rng, snr_db=10, **k):
    p_sig = float(np.mean(audio.astype(np.float64) ** 2)) + 1e-12
    noise = rng.normal(0.0, np.sqrt(p_sig / (10.0 ** (snr_db / 10.0))), size=audio.shape)
    return (audio + noise.astype(np.float32)).astype(np.float32)

def bandlimit(audio, sr, rng, low=120.0, high=6000.0, order=4, **k):
    high = min(high, sr / 2 - 1)
    sos = butter(order, [low, high], btype="band", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)

def clip_dist(audio, sr, rng, drive=0.2, **k):
    return np.clip(audio * (1.0 + drive * 6.0), -1.0, 1.0).astype(np.float32)

_CODECS = {"opus": ("opus", ["-c:a", "libopus"]),
           "mp3":  ("mp3",  ["-c:a", "libmp3lame"]),
           "aac":  ("m4a",  ["-c:a", "aac"])}
def codec_roundtrip(audio, sr, rng, codec="opus", bitrate="24k", **k):
    ext, enc = _CODECS.get(codec, _CODECS["opus"])
    try:
        with tempfile.TemporaryDirectory() as d:
            wi, ec, wo = (os.path.join(d, f) for f in ("in.wav", "e." + ext, "out.wav"))
            sf.write(wi, np.clip(audio, -1, 1).astype(np.float32), sr, subtype="PCM_16")
            subprocess.run(["ffmpeg", "-y", "-i", wi, *enc, "-b:a", bitrate, ec],
                           check=True, capture_output=True)
            subprocess.run(["ffmpeg", "-y", "-i", ec, "-ar", str(sr), "-ac", "1", wo],
                           check=True, capture_output=True)
            y, _ = sf.read(wo, dtype="float32")
        return _fit(np.asarray(y, np.float32), len(audio))
    except Exception as e:
        print(f"    [codec_roundtrip skipped: {e}]")
        return audio

EFFECTS = {
    "reverb_pyroom":   (reverb_pyroom,   "reverb"),
    "reverb_synth":    (reverb_synth,    "reverb"),
    "add_gaussian":    (add_gaussian,    "noise"),
    "bandlimit":       (bandlimit,       "mic"),
    "clip_dist":       (clip_dist,       "mic"),
    "codec_roundtrip": (codec_roundtrip, "mic"),
}

# Identical to benchmark_noisy_kaggle.ipynb -> REALISTIC_PIPELINE
REALISTIC_PIPELINE = [
    ("reverb_pyroom",   dict(rt60=0.45, room=(4.0, 5.0, 3.0))),     # small tiled exam room
    ("add_gaussian",    dict(snr_db=PROFILE_REF_SNR)),              # cheap-mic self-noise -> swept to S
    ("bandlimit",       dict(low=120.0, high=6000.0)),              # cheap-mic rolloff
    ("clip_dist",       dict(drive=0.15)),                          # mild overdrive
    ("codec_roundtrip", dict(codec="opus", bitrate="20k")),         # VoIP / cheap capture
]

def apply_profile(audio, sr, pipeline, snr_value, rng, ref_snr=PROFILE_REF_SNR):
    """Apply a clinic pipeline; 'noise' effects shift by (snr_value - ref_snr)."""
    x = np.asarray(audio, np.float32).copy()
    offset = snr_value - ref_snr
    for name, params in pipeline:
        fn, cat = EFFECTS[name]
        p = dict(params)
        if cat == "noise":
            p["snr_db"] = p.get("snr_db", ref_snr) + offset
        try:
            x = fn(x, sr, rng, **p)
        except Exception as e:
            print(f"    [warn] {name} failed: {e}")
    return x.astype(np.float32)

def pipe_str(pipeline):
    return " -> ".join(n for n, _ in pipeline)


# ── FLEURS streaming (same as the peer samplers) ─────────────────
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
    print(f"Fetched {len(base)} base FLEURS clips -> {len(base)*len(SNR_DB)} {PROFILE_NAME} clips")
    print(f"Pipeline: {pipe_str(REALISTIC_PIPELINE)}")
    rng = np.random.default_rng(1234)

    rows, n = [], 0
    for b in base:
        for snr in SNR_DB:
            noisy = apply_profile(b["array"], b["sr"], REALISTIC_PIPELINE, snr, rng)
            out_wav = os.path.join(OUT_DIR, f"clip_{n:02d}_snr{snr}.wav")
            sf.write(out_wav, np.clip(noisy, -1.0, 1.0), b["sr"], subtype="PCM_16")
            rows.append({
                "clip_id": n,
                "filename": os.path.basename(out_wav),
                "source_file": b["file"],
                "reference": b["reference"],
                "duration_sec": round(len(noisy) / b["sr"], 3),
                "sample_rate": b["sr"],
                "snr_db": snr,
                "noise_type": PROFILE_NAME,
                "pipeline": pipe_str(REALISTIC_PIPELINE),
            })
            print(f"  [{n}] {b['file']}  snr={snr}dB  {len(noisy)/b['sr']:.1f}s")
            n += 1

    with open(os.path.join(OUT_DIR, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {n} {PROFILE_NAME} clips + metadata.csv -> {OUT_DIR}")


if __name__ == "__main__":
    main()
