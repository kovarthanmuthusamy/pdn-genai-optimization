# Multi-Input Multi-Modal VAE Architecture

> 3-modality VAE — Heatmap (1×64×64) + Occupancy (52D) + Impedance (2×231)  
> Latent: **32D** (heatmap) + **32D** (occupancy) + **20D** (impedance) + **48D** (shared PoE) = **132D total**

---

## 1. Encoder — Product of Experts

Three modality branches encode independently. Each branch predicts its own private latent and a shared latent estimate. Shared estimates are fused via precision-weighted Product of Experts (PoE).

```mermaid
graph TB
    INPUT["Multi-Modal Inputs"]
    INPUT --> H["Heatmap\n1×64×64"]
    INPUT --> O["Occupancy\n52D binary"]
    INPUT --> I["Impedance\n2×231"]

    H --> HENC["Conv2d 1→16→32→64\nFC 4096→512→64"]
    O --> OENC["Linear 52→128→256→128→64"]
    I --> IENC["Conv1d 2→16→32→64→128\nFC 1920→256→64"]

    HENC --> HPF["Heatmap Features 64D"]
    OENC --> OPF["Occupancy Features 64D"]
    IENC --> IPF["Impedance Features 64D"]

    HPF --> HZP["Private z_h: mu/logvar → 32D"]
    OPF --> OZP["Private z_o: logits → 32D\nBinary Concrete"]
    IPF --> IZP["Private z_i: mu/logvar → 20D"]

    HPF --> HS["Shared: mu/logvar → 48D"]
    OPF --> OS["Shared: mu/logvar → 48D"]
    IPF --> IS["Shared: mu/logvar → 48D"]

    HS --> POE["Product of Experts\n1/sigma2_comb = sum(1/sigma2_i)"]
    OS --> POE
    IS --> POE

    POE --> ZS["Shared z_s: 48D"]

    HZP --> CAT["Concatenate"]
    OZP --> CAT
    IZP --> CAT
    ZS  --> CAT

    CAT --> Z["Latent z: 132D\n= 32 + 32 + 20 + 48"]
```

**Details:**
- **Heatmap encoder:** Conv2d (1→16→32→64, stride 2) + Flatten + FC → 64D features → private 32D (Gaussian reparameterization) + shared 48D
- **Occupancy encoder:** 4-layer MLP (52→128→256→128→64) → private 32D (Binary Concrete / straight-through) + shared 48D
- **Impedance encoder:** Conv1d on dual-channel input (raw + derivative) → FC → 64D → private 20D + shared 48D
- **PoE prior:** Unit Gaussian prior included as 4th expert

---

## 2. Decoder

Latent z is split, each private slice concatenated with the shared slice, and fed to its own decoder head.

```mermaid
graph TB
    Z["Latent z: 132D"]
    Z --> SPLIT["Split\nz_h=32, z_o=32, z_i=20, z_s=48"]

    SPLIT --> HD["Heatmap decoder\nconcat 32+48 = 80D\nFC 80→256→4096\nReshape + ConvTranspose → 1×64×64"]
    SPLIT --> OD["Occupancy decoder\nconcat 32+48 = 80D\nFC 80→256→512→256→128→52\nSigmoid"]
    SPLIT --> ID["Impedance decoder\nconcat 20+48 = 68D\nFC 68→256→512→462\nReshape → 2×231"]

    HD --> HOUT["Heatmap output\n1×64×64"]
    OD --> OOUT["Occupancy output\n52D binary"]
    ID --> IOUT["Impedance output\n2×231"]
```

---

## 3. Training Pipeline

```mermaid
graph TB
    DATA["Training batch\nHeatmap + Occupancy + Impedance\nbatch_size = 64"]

    DATA --> ENC["Encoder\n3 independent branches"]
    ENC --> POE["PoE Fusion\nshared latent space"]
    POE --> Z["z = 132D"]
    Z --> DEC["Decoder\n3 independent heads"]

    DEC --> PH["Heatmap prediction"]
    DEC --> PO["Occupancy prediction"]
    DEC --> PI["Impedance prediction"]

    PH --> LH["L_heatmap\nMSE"]
    PO --> LO["L_occupancy\nBCE with logits"]
    PI --> LI["L_impedance\nMAE"]
    Z  --> LK["L_KL\nKL divergence"]

    LH --> LTOT["L_total = L_h + L_o + L_i + beta * L_KL\nbeta annealed 0.0 → 0.01 over 200 epochs"]
    LO --> LTOT
    LI --> LTOT
    LK --> LTOT

    LTOT --> OPT["Adam  lr=1e-4\n300 epochs"]
```

---

## 4. Loss Functions

```mermaid
graph TB
    subgraph HLOSS["Heatmap Loss"]
        HP["Predicted 1×64×64"] --> HMSE["MSE"]
        HT["Target 1×64×64"]   --> HMSE
        HMSE --> LH["L_h"]
    end

    subgraph OLOSS["Occupancy Loss"]
        OP["Predicted logits 52D"] --> HBCE["BCE with logits"]
        OT["Target binary 52D"]   --> HBCE
        HBCE --> LO["L_o"]
    end

    subgraph ILOSS["Impedance Loss"]
        IP["Predicted 2×231"] --> HMAE["MAE"]
        IT["Target 2×231"]    --> HMAE
        HMAE --> LI["L_i"]
    end

    subgraph KLLOSS["KL Divergence"]
        MU["mu 132D"]       --> KLC["KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))"]
        LV["logvar 132D"]   --> KLC
        KLC --> LK["L_KL"]
    end

    LH --> TOTAL["L_total = L_h + L_o + L_i + beta * L_KL"]
    LO --> TOTAL
    LI --> TOTAL
    LK --> TOTAL
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Epochs | 300 |
| Batch size | 64 |
| β (KL weight) | 0.0 → 0.01 (annealed epochs 0–200) |
| Heatmap loss | MSE |
| Occupancy loss | BCE with logits |
| Impedance loss | MAE |
