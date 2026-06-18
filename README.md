# EAR: Event-Anchored Retrieval for Dense Video Captioning

This repository provides the pseudocode release of **Event-Anchored Retrieval (EAR)** for dense video captioning.

EAR is designed to improve retrieval-based dense video captioning by aligning query initialization, dynamic retrieval-unit construction, and textual memory readout under a shared event-level semantic prior.

## Overview

Dense video captioning requires localizing and describing multiple events in an untrimmed video. Existing retrieval-based methods introduce external textual memories, but fixed retrieval units may cross event boundaries and inject mixed-event textual evidence.

EAR addresses this issue with two main components:

- **Global Semantic Anchor Bank (GSAB)**  
  Discovers video-internal pseudo-event anchors from frame-level visual features and transfers them to query initialization, boundary calibration, and confidence-aware memory readout.

- **Adaptive Temporal Evolution Modeling (ATEM)**  
  Constructs dynamic retrieval units along local semantic evolution and enhances segment-level representations for more reliable memory retrieval.

## Main Ideas

EAR organizes retrieval-based DVC around a consistent event structure:

1. **Anchor-guided query initialization**  
   GSAB generates pseudo-event proposals and uses them to initialize event queries.

2. **Dynamic semantic partition**  
   ATEM forms retrieval units based on semantic prototype drift instead of fixed temporal windows.

3. **Anchor-calibrated boundary decisions**  
   GSAB labels weakly calibrate temporal boundaries by suppressing splits within the same event anchor and encouraging splits across distant anchors.

4. **Segment representation enhancement**  
   SRE converts each dynamic segment into a stable retrieval query.

5. **Confidence-aware memory readout**  
   GSAB compactness estimates segment reliability and modulates Top-K textual memory fusion.

## Repository Structure

```text
EAR/
├── train.py                    # Training pipeline pseudocode
├── eval.py                     # Evaluation pipeline pseudocode
├── EAR_PSEUDOCODE.md           # High-level framework pseudocode
├── cfgs/
│   ├── yc2_clip_ear.yml
│   └── anet_clip_ear.yml
├── ear/
│   ├── ear_ret_encdec_clip.py  # Main EAR model pseudocode
│   ├── GSAB.py                 # Global Semantic Anchor Bank pseudocode
│   ├── SRE.py                  # Segment Representation Enhancement pseudocode
│   ├── base_encoder.py
│   ├── criterion.py
│   ├── deformable_transformer.py
│   └── matcher.py
└── densevid_eval3/             # Evaluation utilities
