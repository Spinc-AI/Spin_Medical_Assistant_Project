# Persian Speech-to-Text — Benchmark Summary

Comparison of open, locally-runnable Persian STT models for a **clinical (doctor–patient) STT product**. The decision rests on two conditions — **clean speech** and a **realistic clinic-noise** set; three other conditions were tested for context and are summarized briefly at the end.

---

## Datasets

### The two that matter

**1. FLEURS fa_ir — clean (baseline).**
Google FLEURS Persian test split: **871 clips, ~3.7 h**, 16 kHz mono, read literary Persian by native speakers in controlled (near-clean) conditions. Neutral pace, no spontaneous disfluencies. Measures each model's ceiling accuracy and serves as the reference the noisy set is compared against clip-for-clip.

**2. Clinic-realistic noise (`fleurs_noisy_real`) — the decision set.**
Derived from FLEURS: **200 base clips × 5 noise levels = 1,000 clips** (~4.4 h), each passed through a degradation chain that mimics a poor microphone in a small clinical room:

| Stage | Effect | Models the… |
|---|---|---|
| 1 | **Room reverb** (small tiled room, RT60 ≈ 0.45 s) | room echo |
| 2 | **Gaussian noise floor** (swept to target SNR) | cheap-mic electronic self-noise |
| 3 | **Band-limiting** (120–6000 Hz) | cheap-mic frequency rolloff |
| 4 | **Mild clipping** (drive 0.15) | over-driven/close-talked mic |
| 5 | **Opus codec round-trip @ 20 kbps** | VoIP / cheap-capture compression |

Reverb and the mic/codec stages are applied at fixed strength; only the **Gaussian floor varies** across the **SNR sweep of 20 / 15 / 10 / 5 / 0 dB** (200 clips each). Higher dB = cleaner; **20 dB** ≈ mild, **10 dB** ≈ clearly noisy, **0 dB** ≈ noise as loud as the speech. *(The noise floor is synthetic Gaussian — not recorded crowd/cough/HVAC ambience — so the profile models mic + room + codec degradation. A harsher worst-case profile exists but was not run.)*

### Metrics, in brief
**WER/CER** = word/character error rate (lower better). **ΔWER** = clean→noisy degradation on the same clips (smaller = more robust). **% catastrophic** = clips with WER > 0.5 (largely unusable). **% hallucinated** = clips with confident fabrication (the key medical-safety metric). **95% CI** = bootstrap confidence interval on WER (non-overlap = a real difference). **Sub/Ins/Del** = error type — insertions ≈ fabrication, deletions ≈ dropped speech. **VRAM/RTF** = GPU memory and speed (RTF < 1 = faster than real-time).

---

## Results — clean vs. clinic-realistic (N: clean 871, clinic 1000)

Ordered by clinic-noise WER.

| # | Model | Clean WER | Clinic WER | ΔWER | Clinic 95% CI | % catastr. | % halluc. |
|---|---|---|---|---|---|---|---|
| 1 | **seamless-m4t-v2-large** | **0.107** | **0.315** | **+0.207** | [0.29–0.34] | **0.18** | 0.012 |
| 2 | whisper-persian-v4 (nezamisafa) | 0.137 | 0.484 | +0.348 | [0.46–0.51] | 0.41 | **0.004** |
| 3 | hf-seamless-m4t-medium | 0.134 | 0.495 | +0.361 | [0.46–0.54] | 0.31 | 0.046 |
| 4 | persian-whisper-large-v3 (Halakoo) | 0.191 | 0.601 | +0.410 | [0.58–0.64] | 0.57 | 0.006 |
| 5 | mms-1b-all | 0.162 | 0.610 | +0.448 | [0.60–0.63] | 0.63 | 0.000 |
| 6 | mms-1b-fl102 ⚠️ in-domain | 0.146 | 0.623 | +0.477 | [0.61–0.64] | 0.62 | 0.000 |
| 7 | vhdm/whisper-large-fa-v1 | 0.150 | 0.638 | +0.488 | [0.59–0.70] | 0.52 | 0.029 |
| 8 | wav2vec2-large-xlsr-53 | 0.240 | 0.676 | +0.436 | [0.67–0.69] | 0.77 | 0.000 |
| 9 | openai/whisper-large-v3 | 0.198 | 0.716 | +0.518 | [0.67–0.80] | 0.54 | 0.035 |
| 10 | openai/whisper-large-v3-turbo | 0.204 | **1.088** | +0.883 | [1.01–1.22] | 0.60 | **0.114** |

---

## Findings

1. **`seamless-m4t-v2-large` wins both conditions — decisively under noise.** #1 clean (0.107) and #1 clinic (0.315 — 0.17 WER clear of #2, non-overlapping CI on N=1000), with the **smallest degradation** (ΔWER +0.207), **lowest catastrophic rate** (18%), best CER (0.178), and low hallucination (1.2%). It is the base to build the clinical system on (≈11–15 GB VRAM, ~8× real-time).

2. **The ranking reorders under noise, driven by hallucination-resistance.** Persian fine-tunes that don't fabricate rise (nezamisafa, Halakoo); those that do collapse (vhdm, turbo, large-v3). Clean accuracy alone does not predict clinic performance.

3. **The error *type* is the safety story.** Under noise, `seamless-v2` fails almost only by **substitution** (low insertions/deletions) — the recoverable kind. **turbo collapses by fabrication** (WER > 1.0, 11% hallucinated), and `whisper-large-v3` hallucinates once the mic/codec stage hits it. The CTC models (`mms-1b-all`, `wav2vec2`) **never fabricate** but drop speech and score poorly (≥ 0.61 WER) — a "fails-safely" reference, not a primary engine.

4. **Lightweight runner-up: `whisper-persian-v4` (nezamisafa)** — #2 under noise, the lowest hallucination of the field (0.4%), on just **3.4 GB VRAM**. The value pick if Seamless's footprint is too large (≈0.17 WER step-down).

5. **Avoid for clinical use: `turbo` and `vhdm/whisper-large-fa-v1`** — both hallucinate under noise, and confident fabrication (inventing a drug name, dropping a negation) is disqualifying for medical use regardless of clean accuracy.

### Caveats
- **Not production-ready raw.** Even the winner is **0.315 WER / 18% catastrophic** on clinic noise. A deployable system needs **domain fine-tuning + LLM post-correction (e.g. PartAI Dorna2) + confidence-gated human review** on top of the base STT.
- **Healthy voices only.** The clinic set reproduces the *environment* (reverb, cheap mic, codec — ear-validated) but uses healthy FLEURS readers, so it does **not** capture distressed/pathological patient speech. That gap requires real collected clinical data for final sign-off.

---

## Other conditions tested (brief)

- **PSRB (PartAI) — real noisy + spontaneous Persian.** Reordered the field and put `whisper-large-v3` on top, but its content is largely **theatrical TV/film dialogue** (exaggerated delivery), so it's a poor proxy for the clinical speaking register — used as context, not the decision set.
- **Common Voice fa — real read speech.** For several models it was *easier* than clean FLEURS; effectively a second clean set, not a noise test.
- **Synthetic Gaussian-only (SNR sweep).** Pure additive noise on clean read speech **preserved the clean ranking almost exactly** (Spearman ≈ 1.0), so it added little model-selection signal — it only revealed degradation slopes and the CTC noise-cliff. This is why the **realistic** chain (reverb + mic + codec), not plain Gaussian, is the meaningful clinical benchmark.

---

*Open, locally-runnable models only; all inference FP16 on a single NVIDIA T4 (Kaggle). Persian text normalized with Hazm; metrics via jiwer, sacrebleu, bert-score, sentence-transformers.*
