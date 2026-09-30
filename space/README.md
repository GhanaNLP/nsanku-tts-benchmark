---
title: nsanku TTS Benchmark
emoji: 🔊
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: mit
---

# nsanku TTS Benchmark

Multi-metric Text-to-Speech benchmark for Ghanaian languages. 

Models are evaluated across two complementary dimensions:
1. **Intelligibility (Character Accuracy = 1 - CER)**: Per-language ASR judges transcribe each clip; scored against the sentence read.
2. **Acoustic Naturalness (SpeechBERTScore)**: Precision variant of `microsoft/wavlm-large` layer 6 scored against real recorded human speech references from `ghananlpcommunity/ghana-speech-eval`.

**Overall Ranking:** Ranked by **Composite Score** = `(Accuracy + SpeechBERTScore) / 2`.

- GitHub Repository: [GhanaNLP/nsanku-tts-benchmark](https://github.com/GhanaNLP/nsanku-tts-benchmark)
