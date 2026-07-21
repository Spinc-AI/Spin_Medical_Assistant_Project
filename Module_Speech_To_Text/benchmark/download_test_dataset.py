#!/usr/bin/env python3
'''
Downloads an annotated Persian speech dataset and extracts a small subset
(default 100 clips) for benchmarking STT models.

NOTE on FLEURS: the fa_ir test split is a single ~805 MB Parquet file (one row
group of 871 clips). Parquet can only be read a whole row group at a time, so
the full file must be downloaded once - there is no way to fetch just N clips
over the network. This script therefore:
  1. Downloads the Parquet file ONCE (with a progress bar + resume; re-running
     reuses the local file, it does NOT re-download).
  2. Extracts the first N clips LOCALLY (no extra network):
       - each clip as a 16 kHz mono WAV in <output_dir>/clips/
       - a metadata.tsv mapping each WAV filename to its reference transcript

For a faster, resumable download on a slow connection:
  - High-performance transfer uses the Xet backend (hf_xet). This script
    auto-enables it via HF_XET_HIGH_PERFORMANCE if hf_xet is installed.
  - set HF_TOKEN for higher rate limits (huggingface-cli / login())

To Use as Standalone:
1. Install the required libraries:
   pip install datasets soundfile pyarrow hf_xet
2. Run this file. By default it downloads FLEURS fa_ir test once and extracts
   100 clips into "input/test_data/fleurs_fa".

To Use as a Module:
from download_test_dataset import download_subset
download_subset("fleurs", num_clips=100, output_dir="input/test_data/fleurs_fa")
'''

import io
import os

# Enable high-performance transfer via the Xet backend if hf_xet is installed.
# (huggingface_hub 1.x replaced hf_transfer with Xet; the old
#  HF_HUB_ENABLE_HF_TRANSFER flag is deprecated.)
try:
    import hf_xet  # noqa: F401
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
except ImportError:
    pass

import soundfile as sf
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

DATASETS = {
    "fleurs": {
        "path": "google/fleurs",
        "config": "fa_ir",
        "split": "test",
        "revision": "refs/convert/parquet",
        "audio_key": "audio",
        "text_key": "transcription",
    },
}

def _find_parquet_files(cfg):
    files = list_repo_files(cfg["path"], repo_type="dataset", revision=cfg["revision"])
    matches = sorted(
        f for f in files
        if f.endswith(".parquet")
        and f.startswith(f"{cfg['config']}/{cfg['split']}/")
    )
    if not matches:
        raise RuntimeError(
            f"No parquet files found for {cfg['path']} {cfg['config']}/{cfg['split']}"
        )
    return matches

def download_subset(dataset_name, num_clips, output_dir):
    '''
    Downloads a Persian speech dataset (once) and extracts the first `num_clips`
    clips to disk as WAV files plus a metadata.tsv transcript map.
    Args:
        dataset_name (str): Currently supports "fleurs".
        num_clips (int): Number of clips to extract locally.
        output_dir (str): Directory where raw/, clips/ and metadata.tsv are written.
    '''
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from {list(DATASETS)}.")

    cfg = DATASETS[dataset_name]
    raw_dir = os.path.join(output_dir, "raw")
    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)

    # --- Step 1: download the parquet file(s) once (resumable, with progress bar)
    parquet_files = _find_parquet_files(cfg)
    print(f"Downloading {len(parquet_files)} parquet file(s) for "
          f"{cfg['path']} ({cfg['config']}/{cfg['split']})...")
    print("(One-time download; re-running reuses the local copy.)\n")

    local_paths = []
    for remote in parquet_files:
        expected = os.path.join(raw_dir, *remote.split("/"))
        if os.path.exists(expected):
            # Already downloaded (e.g. manually via a download manager) - reuse it
            # and skip the network entirely.
            print(f"Found existing file, skipping download: {expected}")
            local_paths.append(expected)
            continue
        local = hf_hub_download(
            repo_id=cfg["path"],
            filename=remote,
            repo_type="dataset",
            revision=cfg["revision"],
            local_dir=raw_dir,
        )
        local_paths.append(local)

    # --- Step 2: extract the first num_clips clips locally (no network)
    print(f"\nExtracting first {num_clips} clips...")
    metadata_path = os.path.join(output_dir, "metadata.tsv")
    saved = 0
    with open(metadata_path, "w", encoding="utf-8") as meta:
        meta.write("filename\ttranscript\n")
        for local in local_paths:
            if saved >= num_clips:
                break
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(
                batch_size=64, columns=[cfg["audio_key"], cfg["text_key"]]
            ):
                for row in batch.to_pylist():
                    if saved >= num_clips:
                        break
                    audio_bytes = row[cfg["audio_key"]]["bytes"]
                    transcript = (row[cfg["text_key"]] or "").strip()
                    transcript = transcript.replace("\t", " ").replace("\n", " ")

                    arr, sr = sf.read(io.BytesIO(audio_bytes))
                    filename = f"{dataset_name}_{saved:04d}.wav"
                    sf.write(os.path.join(clips_dir, filename), arr, sr)
                    meta.write(f"{filename}\t{transcript}\n")

                    saved += 1
                    print(f"[{saved}/{num_clips}] saved {filename}")
                if saved >= num_clips:
                    break

    print(f"\nDone. Saved {saved} clips to: {clips_dir}")
    print(f"Transcripts written to: {metadata_path}")
    return saved

if __name__ == "__main__":

    root_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_name = "fleurs"
    num_clips = 100

    output_dir = os.path.join(root_dir, "input", "test_data", f"{dataset_name}_fa")

    download_subset(dataset_name, num_clips, output_dir)
