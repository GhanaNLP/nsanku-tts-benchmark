> **Note:** the former "benchmark 1" (CER on ghana-sentences) and "benchmark 2" (SpeechBERTScore) are now one benchmark on ghana-speech-eval; CER and SpeechBERTScore are both computed per clip and combined as a composite. Mentions of "benchmark 1/2" below are historical.

# Methodology

## What benchmark 2 measures

Every sample is a row of [`ghananlpcommunity/ghana-speech-eval`](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech-eval):
a real recording of a sentence, read aloud, **with its transcript**. A TTS model
is asked to read that transcript, and the recording of it becomes the reference.

```
ghana-speech-eval row i          TTS model            reference = row i's audio
  text  ──────────────────────▶  synthesise  ──────▶  wav  vs  wav
```

This is what benchmark 1 cannot do. Benchmark 1 draws its sentences from
`ghanaopenai/ghana-sentences` while its only real recordings are twelve prompt
clips, one per language, taken from a deliberately different corpus so nothing
leaks. It therefore has no per-sample reference, and scores intelligibility with
CER instead.

## The metric

SpeechBERTScore ([Saeki et al., Interspeech 2024](https://arxiv.org/abs/2401.16812))
compares generated speech to a real reference in a frozen self-supervised
encoder's feature space:

```
Z_hat = Encoder(X_hat)      Z = Encoder(X)

SpeechBERTScore = (1 / N_gen) * sum_i max_j cos( z_hat_i , z_j )      (Eq. 6)
```

Each generated frame is matched to its most similar reference frame, so the two
utterances need not be the same length or time-aligned. Features are
L2-normalised first, so the score is a cosine similarity in `[0, 1]`: **1.0** is
an identical clip, and **higher is better**.

We use the **precision** variant, which the paper found correlated best with
human ratings (Appendix A: precision 0.545, F1 0.530, recall 0.495, utterance
level SRCC).

Encoder and layer: `microsoft/wavlm-large`, layer 6. See
[encoder-selection.md](encoder-selection.md) for the sweep behind that.

## Implementation notes

**Sample rate.** Features are computed at 16 kHz, the encoder's native rate.
Synthesised clips are resampled from whatever their model emits, so a 48 kHz
output is not silently penalised for being at a different rate.

**Reference audio** is extracted once from the dataset's parquet configs to
`references/{iso}/{row:05d}.wav` and reused; scoring a 200-sample run should
not re-read a 15 GB parquet.

**Chunking.** The greedy match is computed in blocks of 4096 generated frames so
a long clip never materialises a full `[T_gen, T_ref]` matrix. Chunk size
changes only the float32 summation order, not the value (measured delta 3e-8).

## Determinism

`benchmark/speechbertscore.py` asserts, before scoring anything, that encoding
the same audio twice gives bit-identical features and a self-similarity of
exactly 1.0.

This is not ceremony. An earlier implementation encoded through a pretraining
path that applied SpecAugment-style masking, so repeated encodes of the same
audio differed by up to 7.9 and self-similarity was 0.91. The metric still
*looked* like it worked — it produced plausible, stable-looking, wrong numbers,
and every layer decision made on top of it was compromised. The fix was to
bypass the pretraining path and call the encoder directly.

## Ref-mode models: the prompt is never the reference

Several models are prompted with a real recording to clone a voice. In
benchmark 2 that creates a trap: if such a model were prompted with the very
clip SpeechBERTScore scores it against, the metric would measure how well the
model **copies the clip it was handed**, not how well it synthesises. Such a
model would top the leaderboard for the wrong reason.

So the prompt clip is drawn from a **fixed offset of 500 rows** away
(`PROMPT_ROW_OFFSET`). The offset is fixed rather than random so a re-run prompts
identically and stays comparable. A recipe may still pin a prompt clip, but
`reference_clip()` refuses an override that collides with the row being scored
and falls back to the offset row, logging a warning.

The shipped recipes all set `REFERENCE_CLIP = None`, so this is a guard against
a future recipe rather than a fix for a present one.

## What SpeechBERTScore does and does not catch

Measured on real ghana-speech-eval audio at the selected configuration. Each
figure is the fraction by which a perturbation moves the score, so bigger is
"the metric notices more".

| perturbation | caught? | notes |
|---|---|---|
| wrong words | strongly | content sensitivity ~0.32 at the selected layer |
| added noise | strongly | noise sensitivity ~0.32 at the selected layer |
| speaking rate | **barely** | ~0.014 — a 15% time-stretch moves the score very little |
| a different speaker | partly | voice similarity is entangled with the metric |

The speaking-rate insensitivity is a real limitation and worth stating plainly:
**SpeechBERTScore is not a prosody metric.** It will not reliably rank two
models that read the same text at visibly different speeds. Since benchmark 2
has no CER, it also does not directly penalise a garbled utterance that happens
to stay near the reference in feature space.

The metric is used on its own here because it is the reference-free companion
to intelligibility, not instead of it. Benchmark 1's CER leaderboard and
benchmark 2's SpeechBERTScore leaderboard are complementary, and a model that
does badly on one and well on the other is exactly the interesting case.

## Reproducibility

`data/manifest_{iso}.json` records the exact rows, their texts and durations
used for a run, plus the source dataset and prompt offset. Scores are keyed by
row index, so re-running only scores rows not already present, and bumping
`NUM_SAMPLES` extends a leaderboard without invalidating it.

## Two-stage run

Synthesis and scoring are separate because re-scoring must not require
re-synthesising, and because a TTS model and the encoder rarely both fit on one
GPU:

```
python -m benchmark.evaluate synthesize --iso twi_asante
python -m benchmark.evaluate score      --iso twi_asante
```

Clips on disk are reused unless `--force` is passed.
