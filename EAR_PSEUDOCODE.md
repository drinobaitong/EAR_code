# EAR Pipeline Pseudocode

This document summarizes the main logic of the EAR codebase.

## Main training loop (`train.py`)

```text
load configuration and datasets
build EAR model, criterion, optimizer, and schedulers
for each epoch:
    train one epoch on caption/localization losses
    if validation is enabled:
        run evaluation on the validation split
        compute captioning and localization metrics
        update best checkpoint according to the selected score
save logs, predictions, and training metadata
```

## Evaluation loop (`eval.py`)

```text
load configuration, validation dataset, and checkpoint
build EAR model and load compatible weights
for each validation batch:
    prepare visual features and textual memory
    run EAR forward inference
    decode event proposals and captions
aggregate predictions
compute DVC, grounding, and paragraph-level metrics when enabled
write evaluation logs and analysis statistics
```

## EAR model (`ear/ear_ret_encdec_clip.py`)

```text
input: video features, masks, timestamps, and textual memory bank
extract or reuse CLIP frame features
build GSAB anchors from frame-level CLIP features
    cluster frames into pseudo-event anchors
    produce anchor labels, proposals, and compactness confidence
initialize decoder queries with GSAB proposals plus learnable queries
run ATEM retrieval branch
    apply DSP to create dynamic semantic segments
        smooth frame features
        maintain long-term and short-term segment prototypes
        compare prototype drift with GSAB-calibrated boundary threshold
        merge too-short segments
    apply SRE to aggregate each segment into a retrieval-ready feature
    retrieve Top-K sentence memories for each segment
    use GSAB confidence to weight retrieved memories
encode visual features and retrieved text features
run Transformer decoder for event localization and caption generation
output losses during training or decoded predictions during evaluation
```

## GSAB utilities (`ear/GSAB.py`)

```text
input: frame-level video features and timestamps
cluster frame features into pseudo-event labels
split discontinuous regions with the same label
filter short events
compute temporal proposal centers and durations
compute anchor centers and compactness confidence
return proposals, labels, anchor centers, and confidence scores
```

## SRE module (`ear/SRE.py`)

```text
input: frame features inside one dynamic segment
split features into stable semantic and local response paths
estimate segment-level statistics for semantic modulation
preserve local temporal responses in the detail path
fuse both paths
pool the enhanced sequence into one segment representation
```
