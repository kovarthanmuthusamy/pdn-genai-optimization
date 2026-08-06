---
title: exp001_new_architecture
type: experiment
status: historical
era: gan
tags: [experiment, exp001_new_architecture, historical, era-gan]
---

# exp001_new_architecture

**Lineage:** **exp001_new_architecture** → [[exp002_hyper_parms_change]]
**Status:** historical · **Era:** GAN era (0 Python files)

## Notes (from `experiments/exp001_new_architecture/notes.md`)

# Experiment: exp001_new_architecture

## Goal
Try new architecture of GAN for the PDN optimization

## Changes
Removed condition in the CGAN and converted to normal GAN that will be trained with datasets to produce the distribution but the binary datas in the dataset remains unchanged eg . occ_map,mask (0/1)

## Results
yet to try

## Decision
GAN whole strcture - fusion model / conditions/ only noise in the generator input

## Checkpoints

`epoch_10.pt`, `epoch_100.pt`, `epoch_110.pt`, `epoch_120.pt`, `epoch_130.pt`, `epoch_140.pt`, `epoch_150.pt`, `epoch_160.pt`, `epoch_170.pt`, `epoch_180.pt`, `epoch_190.pt`, `epoch_20.pt`, `epoch_200.pt`, `epoch_30.pt`, `epoch_40.pt`, `epoch_50.pt`, `epoch_60.pt`, `epoch_70.pt`, `epoch_80.pt`, `epoch_90.pt`, `last.pt`
