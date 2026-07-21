"""
synthetic_worst_sample.py — build 10 clinic-WORST-CASE noisy clips from
google/fleurs (fa_ir) WITHOUT downloading the full dataset.

10 FINAL clips = 2 base FLEURS clips x 5 SNRs [20, 15, 10, 5, 0] dB.
Profile (matches benchmark_noisy_kaggle.ipynb -> WORST_PIPELINE):
    room reverb -> babble (MUSAN) -> cough/breath events (ESC-50) -> HVAC hum
    -> monitor beeps -> Gaussian -> band-limit -> mild clip -> Opus codec.
The SNR sweep shifts every additive-noise component together by (S - PROFILE_REF_SNR),
so at PROFILE_REF_SNR(=10) the mix equals the tuned levels below; higher S = cleaner,
lower S = louder. Reverb and the mic chain are fixed channel effects.

Noise banks (babble + cough) are discovered from, in order:
    $NOISE_BANKS_DIR  ->  /kaggle/input  ->  ./noise_banks
Attach MUSAN (babble) + ESC-50 (cough/breath) for the full worst case. If a bank is
missing, that layer is skipped (with a warning) and the rest of the profile still runs.

Minimum download: test.tsv (~0.71 MB) + leading ~1.5 MB of test.tar.gz (2 clips).

Requires:  pip install requests soundfile numpy scipy
           optional: pyroomacoustics (realistic reverb; falls back to synthetic IR)
           optional: ffmpeg on PATH (codec round-trip; skipped if absent)
           MUSAN + ESC-50 banks for the babble / cough layers
Output:    samples/synthetic_worst/clip_00_snr20.wav ... clip_09_snr0.wav  +  metadata.csv
"""
import os, io, csv, tarfile, subprocess, tempfile
import numpy as np
import soundfile as sf
import requests
from math import gcd
from scipy.signal import resample_poly, butter, sosfilt

REPO            = "google/fleurs"
LANG, SPLIT     = "fa_ir", "test"
N_BASE          = 2
SNR_DB          = [20, 15, 10, 5, 0]
PROFILE_REF_SNR = 10           # sweep midpoint: at this SNR the pipeline == the tuned mix
PROFILE_NAME    = "clinic_worst"
SYNTH_SR        = 16000        # banks are mixed at 16 kHz
OUT_DIR         = os.path.join("samples", "synthetic_worst")

# where to look for the noise banks (first existing wins)
BANK_SEARCH_ROOTS = [os.environ.get("NOISE_BANKS_DIR"), "/kaggle/input", "noise_banks"]

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

def _load_wav(path, sr_target=SYNTH_SR):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1).astype(np.float32)
    if sr != sr_target and len(x):
        g = gcd(int(sr), int(sr_target))
        x = resample_poly(x, sr_target // g, sr // g).astype(np.float32)
    return x


# ── noise banks (MUSAN / ESC-50 / DEMAND) ────────────────────────
class NoiseBank:
    def __init__(self, files, sr):
        self.files = list(files); self.sr = sr
    def _read(self, path):
        try:
            return _load_wav(path, self.sr)
        except Exception:
            return np.zeros(0, np.float32)
    def sample(self, n, rng):
        if not self.files:
            return np.zeros(n, np.float32)
        x = self._read(str(rng.choice(self.files)))
        if len(x) == 0:
            return np.zeros(n, np.float32)
        if len(x) > n:
            s = int(rng.integers(0, len(x) - n + 1)); x = x[s:s + n]
        return _fit(x, n)
    def event(self, rng):
        if not self.files:
            return np.zeros(0, np.float32)
        return self._read(str(rng.choice(self.files)))

def _walk_wavs(root, exts=(".wav", ".flac", ".mp3", ".ogg")):
    if not root or not os.path.isdir(root):
        return []
    return [os.path.join(dp, f) for dp, _, fns in os.walk(root)
            for f in fns if f.lower().endswith(exts)]

def _find_dir(*keys):
    for root in BANK_SEARCH_ROOTS:
        if not root or not os.path.isdir(root):
            continue
        for dp, dns, _ in os.walk(root):
            for d in dns:
                if any(k in d.lower() for k in keys):
                    return os.path.join(dp, d)
    return None

def _subdir(root, name):
    if not root:
        return None
    for dp, dns, _ in os.walk(root):
        for d in dns:
            if d.lower() == name:
                return os.path.join(dp, d)
    return root

def _esc50_bank(root, categories, sr):
    cat = {}
    for dp, _, fns in os.walk(root or ""):
        for f in fns:
            if f.lower().endswith(".csv"):
                try:
                    for r in csv.DictReader(open(os.path.join(dp, f), encoding="utf-8")):
                        if "filename" in r and "category" in r:
                            cat[r["filename"]] = r["category"]
                except Exception:
                    pass
    allw = _walk_wavs(root)
    if cat and categories:
        files = [w for w in allw if cat.get(os.path.basename(w)) in set(categories)]
    else:
        files = allw
    return NoiseBank(files, sr)

_musan  = _find_dir("musan")
_esc    = _find_dir("esc50", "esc-50", "environmental-sound")
_demand = _find_dir("demand")
BANKS = {
    "babble":  NoiseBank(_walk_wavs(_subdir(_musan, "speech")), SYNTH_SR),
    "ambient": NoiseBank(_walk_wavs(_subdir(_musan, "noise")) + _walk_wavs(_demand), SYNTH_SR),
    "cough":   _esc50_bank(_esc, ["coughing", "breathing", "sneezing"], SYNTH_SR),
}


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

def add_noise(audio, sr, rng, bank="ambient", snr_db=15, **k):
    nb = BANKS.get(bank)
    if not nb or not nb.files:
        return audio
    return (audio + scale_to_snr(audio, nb.sample(len(audio), rng), snr_db)).astype(np.float32)

def add_babble(audio, sr, rng, snr_db=15, n_voices=4, bank="babble", **k):
    nb = BANKS.get(bank)
    if not nb or not nb.files:
        return audio
    mix = np.zeros(len(audio), np.float32)
    for _ in range(n_voices):
        mix += nb.sample(len(audio), rng)
    return (audio + scale_to_snr(audio, mix, snr_db)).astype(np.float32)

def add_events(audio, sr, rng, bank="cough", snr_db=10, n_events=2, **k):
    nb = BANKS.get(bank)
    if not nb or not nb.files:
        return audio
    out = audio.copy()
    for _ in range(n_events):
        ev = nb.event(rng)
        if len(ev) == 0:
            continue
        ev = ev[:len(audio)]
        pos = int(rng.integers(0, max(1, len(audio) - len(ev))))
        seg = out[pos:pos + len(ev)]
        ev = scale_to_snr(audio, ev[:len(seg)], snr_db)
        out[pos:pos + len(ev)] += ev
    return out.astype(np.float32)

def synth_hum(audio, sr, rng, snr_db=28, base=50.0, n_harm=4, **k):
    t = np.arange(len(audio)) / sr
    hum = sum(np.sin(2 * np.pi * base * h * t) / h for h in range(1, n_harm + 1))
    return (audio + scale_to_snr(audio, hum.astype(np.float32), snr_db)).astype(np.float32)

def synth_beeps(audio, sr, rng, snr_db=22, freq=1000.0, beep_ms=150, interval=5.0, **k):
    bn = int(sr * beep_ms / 1000.0)
    beep = (np.sin(2 * np.pi * freq * np.arange(bn) / sr) * np.hanning(bn)).astype(np.float32)
    tmpl = np.zeros(len(audio), np.float32); step = max(bn, int(sr * interval))
    for pos in range(0, len(audio) - bn, step):
        tmpl[pos:pos + bn] += beep
    return (audio + scale_to_snr(audio, tmpl, snr_db)).astype(np.float32)

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
    "add_noise":       (add_noise,       "noise"),
    "add_babble":      (add_babble,      "noise"),
    "add_events":      (add_events,      "noise"),
    "synth_hum":       (synth_hum,       "noise"),
    "synth_beeps":     (synth_beeps,     "noise"),
    "bandlimit":       (bandlimit,       "mic"),
    "clip_dist":       (clip_dist,       "mic"),
    "codec_roundtrip": (codec_roundtrip, "mic"),
}

# Identical to benchmark_noisy_kaggle.ipynb -> WORST_PIPELINE
WORST_PIPELINE = [
    ("reverb_pyroom",   dict(rt60=0.45, room=(4.0, 5.0, 3.0))),
    ("add_babble",      dict(snr_db=15, n_voices=4)),               # waiting-room chatter
    ("add_events",      dict(bank="cough", snr_db=8, n_events=2)),  # patient cough/breath
    ("synth_hum",       dict(snr_db=28, base=50.0, n_harm=4)),      # HVAC / mains hum
    ("synth_beeps",     dict(snr_db=22, freq=1000.0, interval=5.0)),# monitor beep
    ("add_gaussian",    dict(snr_db=20)),                           # cheap-mic floor
    ("bandlimit",       dict(low=120.0, high=6000.0)),
    ("clip_dist",       dict(drive=0.15)),
    ("codec_roundtrip", dict(codec="opus", bitrate="20k")),
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
    print("Noise banks:", {k: len(v.files) for k, v in BANKS.items()},
          f"(musan={_musan is not None}, esc50={_esc is not None}, demand={_demand is not None})")
    if not BANKS["babble"].files or not BANKS["cough"].files:
        print("  [WARN] MUSAN/ESC-50 banks not found -> babble/cough layers skipped. "
              "Set NOISE_BANKS_DIR (or run on Kaggle with the datasets attached) for the full worst case.")

    base = fetch_base_clips(N_BASE)
    print(f"Fetched {len(base)} base FLEURS clips -> {len(base)*len(SNR_DB)} {PROFILE_NAME} clips")
    print(f"Pipeline: {pipe_str(WORST_PIPELINE)}")
    rng = np.random.default_rng(1234)

    rows, n = [], 0
    for b in base:
        for snr in SNR_DB:
            noisy = apply_profile(b["array"], b["sr"], WORST_PIPELINE, snr, rng)
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
                "pipeline": pipe_str(WORST_PIPELINE),
            })
            print(f"  [{n}] {b['file']}  snr={snr}dB  {len(noisy)/b['sr']:.1f}s")
            n += 1

    with open(os.path.join(OUT_DIR, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {n} {PROFILE_NAME} clips + metadata.csv -> {OUT_DIR}")


if __name__ == "__main__":
    main()
