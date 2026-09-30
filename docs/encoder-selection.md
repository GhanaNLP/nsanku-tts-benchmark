# Encoder and layer selection

The SpeechBERTScore definition names an encoder and a layer, and the paper's
numbers are only comparable to a run that uses the same ones. Rather than take
either on faith, both were measured on this benchmark's own data.

**Selected: `microsoft/wavlm-large`, layer 6, precision variant.**

## How the sweep was run

Three candidate encoders, on the same 6 languages (Asante Twi, Ewe, Dagaare,
Gonja, Fante, Kasem) and the same real ghana-speech-eval rows. 25 samples per
language for the 300M model, 20 for the others.

Each candidate layer was scored on three perturbations of a real recording:

| measure | how it is built | what a good value means |
|---|---|---|
| **content sensitivity** | score a *different* sentence, `1 - score` | large: the metric notices wrong words |
| **speed penalty** | resample the same clip ~15% faster, `1 - score` | small: the metric is not just a speed check |
| **noise sensitivity** | add background noise, `1 - score` | large: the metric notices degraded audio |

A layer that scores high on all three is separating speech from non-speech
rather than one particular artefact.

## Results

| encoder | params | layers | best layer | content | speed penalty | noise |
|---|---|---|---|---|---|---|
| omniASR W2V 300M | 317M | 24 | 7 | 0.245 | 0.006 | 0.200 |
| omniASR W2V 1B | 966M | 48 | 5 | 0.292 | 0.010 | **0.330** |
| **wavlm-large** | 315M | 24 | **6** | **0.317** | 0.014 | 0.317 |

Full wavlm-large layer curve:

| layer | 2 | 3 | 4 | 5 | **6** | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| content | 0.205 | 0.253 | 0.258 | 0.279 | **0.317** | 0.306 | 0.293 | 0.299 | 0.312 | 0.294 | 0.310 |
| noise | 0.320 | 0.348 | 0.354 | 0.333 | 0.317 | 0.276 | 0.239 | 0.218 | 0.218 | 0.202 | 0.191 |
| speed pen. | 0.014 | 0.015 | 0.015 | 0.014 | **0.014** | 0.013 | 0.011 | 0.009 | 0.009 | 0.009 | 0.008 |

Per-language content sensitivity at layer 6 is tight — 0.295 to 0.317 across
the six languages — so the choice is not being driven by one easy language.

## Why wavlm-large, and not the multilingual model

The obvious candidate was a multilingual encoder trained on African languages,
on the reasoning that it should understand them better. Measured, it does not
win on the property that matters most here.

**omniASR-W2V-1B is a genuine tradeoff, not a win.** It leads on noise (0.330 vs
0.317) but trails on content (0.292 vs 0.317). Content is the discriminating
property for a benchmark whose whole point is reading a *known* sentence: if the
metric cannot tell correct words from wrong ones, it cannot rank TTS models.
Noising a clip is a different job.

**wavlm-large was also the more conservative choice.** It is the paper's own
default encoder, so its numbers are directly comparable to published
SpeechBERTScore values, and it is a third of the size. Lower parameter count is
also better for a public benchmark: anyone can rerun the metric on a single
consumer GPU.

**The multilingual option stays available.** `ENCODER_MODEL` and
`ENCODER_LAYER` are environment variables, so scoring the same clips under
omniASR-W2V-1B is a two-line change, and doing so is a reasonable way to check
whether a conclusion is an artefact of the encoder.

## Honest caveats

**The layer curve is a plateau, not a peak.** WavLM layers 6 to 12 sit between
0.293 and 0.317 on content sensitivity, a spread comparable to the sampling
error at 20 samples per language. Layer 6 is picked because it maximises the
content measure, which is the criterion that matters, and because it sits in the
region the original paper defaults to — **not** because it is meaningfully
better than layer 10. Reporting it as a sharp optimum would be false precision.

**20 samples per language is a small pilot.** Enough to rule out a bad encoder
and to find a plausible layer. Not enough to claim layer 6 beats layer 10. If
the distinction ever matters for a ranking, re-run the sweep at 60+ samples.

**The composite ranking is not a leaderboard criterion.** It was used only to
locate a candidate layer per encoder. Scores from different encoders are not
comparable to each other and are not mixed anywhere in this repo.

**Sensitivity is measured, validity is not.** These numbers show the metric
*responds* to degradation. Whether its ranking matches human preference on
Ghanaian languages is a separate question this sweep does not answer — the
paper's human-correlation study was on English and Japanese.
