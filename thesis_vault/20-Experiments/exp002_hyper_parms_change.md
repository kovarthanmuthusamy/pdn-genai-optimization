---
title: exp002_hyper_parms_change
type: experiment
status: historical
era: unknown
tags: [experiment, exp002_hyper_parms_change, historical, era-unknown]
---

# exp002_hyper_parms_change

**Lineage:** [[exp001_new_architecture]] → **exp002_hyper_parms_change** → [[exp003]]
**Status:** historical · **Era:** era undetermined (0 Python files)

## Notes (from `experiments/exp002_hyper_parms_change/notes.md`)

# Experiment: exp002_hyper_parms_change

## Goal
Critic was very ocsillative

## Changes
Lambda ,critic_iter,lr_d,batch_size , binary values in the dataset ar now changed to differential approximation for avoiding graient explosion or vanishing. -- using sigmoid function on binary values

## Results
observe

## Decision

## Metrics artifacts

`experiments/exp002_hyper_parms_change/metrics/`

- `loss.csv`
- `summary.json`
- `violation_ratio.csv`

## Checkpoints

`epoch_10.pt`, `epoch_100.pt`, `epoch_110.pt`, `epoch_120.pt`, `epoch_130.pt`, `epoch_140.pt`, `epoch_150.pt`, `epoch_160.pt`, `epoch_170.pt`, `epoch_180.pt`, `epoch_190.pt`, `epoch_20.pt`, `epoch_200.pt`, `epoch_30.pt`, `epoch_40.pt`, `epoch_50.pt`, `epoch_60.pt`, `epoch_70.pt`, `epoch_80.pt`, `epoch_90.pt`
