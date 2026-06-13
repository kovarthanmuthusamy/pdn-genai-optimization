#!/usr/bin/env python3
"""
VAE Architecture Viewer - Generates standalone HTML file with Mermaid diagrams
"""

import os
import base64

def encode_image(img_path):
    """Encode image file to base64 data URI."""
    try:
        with open(img_path, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        return f"data:image/png;base64,{data}"
    except FileNotFoundError:
        return ""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>VAE Encoder, Decoder, and Losses</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #333;
        }
        .container {
            max-width: 1500px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.4em;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .tabs {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
            background: #ecf0f1;
            border-radius: 10px;
            padding: 5px;
            flex-wrap: wrap;
        }
        .tab {
            padding: 14px 18px;
            margin: 5px;
            border: none;
            background: transparent;
            cursor: pointer;
            border-radius: 8px;
            font-size: 0.95em;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .tab.active {
            background: #3498db;
            color: white;
            box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        }
        .tab:hover:not(.active) {
            background: #bdc3c7;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .info-box {
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 25px;
        }
        .info-box ul {
            margin: 0;
            padding-left: 20px;
        }
        .footer {
            text-align: center;
            color: #7f8c8d;
            margin-top: 30px;
            font-size: 0.9em;
        }
        .mermaid {
            background: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.05);
            margin-bottom: 25px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Multi-Input Multi-Modal VAE with Product of Experts</h1>
        <p class="subtitle">3-modality VAE (Heatmap + Occupancy + Impedance) using Product of Experts fusion for the shared latent space. Private latent dimensions: Heatmap 32D, Occupancy 32D, Impedance 20D. Shared dimension: 48D. Total latent: 132D.</p>

        <div class="tabs">
            <button class="tab active" onclick="showTab(event, 'encoder')">🧩 Encoder</button>
            <button class="tab" onclick="showTab(event, 'decoder')">🎯 Decoder</button>
            <button class="tab" onclick="showTab(event, 'pipeline')">📊 Data Pipeline</button>
            <button class="tab" onclick="showTab(event, 'losses')">⚖️ Loss Functions</button>
            <button class="tab" onclick="showTab(event, 'results')">🖼️ Results</button>
        </div>

        <div id="encoder" class="tab-content active">
            <div class="info-box">
                <h2>🧩 Product of Experts (PoE) Multi-Modal Encoder</h2>
                <p>Four branches (heatmap, occupancy, impedance, max_value) encode independently. Each modality predicts the shared latent space with uncertainty, combined via precision-weighted Product of Experts fusion.</p>
            </div>

            <div class="mermaid">
flowchart TB
    %% Input Layer
    INPUT[🎯 MULTI-MODAL INPUTS]
    
    INPUT --> H_IN[Heatmap<br/>1×64×64]
    INPUT --> O_IN[Occupancy<br/>52-D vector]
    INPUT --> I_IN[Impedance<br/>2×231<br/>raw + derivative]
    
    %% Encoder Branches
    H_IN --> H_ENC["🔥 Heatmap Encoder<br/>Conv2d: 1→16→32→64<br/>FC: 4096→512→64"]
    O_IN --> O_ENC["🗺️ Occupancy Encoder<br/>Linear: 52→128→256→128→64"]
    I_IN --> I_ENC["📊 Impedance Encoder<br/>Conv1d: 2→16→32→64→128<br/>FC: 1920→256→64"]
    
    H_ENC --> H_FEAT["Heatmap Features<br/>64-D"]
    O_ENC --> O_FEAT["Occupancy Features<br/>64-D"]
    I_ENC --> I_FEAT["Impedance Features<br/>64-D"]
    
    %% Private Latent Spaces
    H_FEAT --> H_PRIV["Private Heatmap<br/>μ_h, logvar_h<br/>32-D"]
    O_FEAT --> O_PRIV["Private Occupancy<br/>logits → Binary Concrete<br/>32-D"]
    I_FEAT --> I_PRIV["Private Impedance<br/>μ_i, logvar_i<br/>20-D"]
    
    %% Shared Latent Spaces for PoE
    H_FEAT --> H_SHARED["Shared Prediction<br/>μ_h, logvar_h<br/>48-D"]
    O_FEAT --> O_SHARED["Shared Prediction<br/>μ_o, logvar_o<br/>48-D"]
    I_FEAT --> I_SHARED["Shared Prediction<br/>μ_i, logvar_i<br/>48-D"]
    
    %% Product of Experts Fusion
    H_SHARED --> POE["⚡ Product of Experts Fusion<br/>precision:  1/σ²_comb = Σ(1/σ²_i) + 1/σ²_prior<br/>mean:  μ_comb = σ²_comb · Σ(μ_i/σ²_i)<br/>unit Gaussian N(0,1) included as 4th expert"]
    O_SHARED --> POE
    I_SHARED --> POE
    
    POE --> SHARED["Combined posterior<br/>μ_comb, σ²_comb  —  48-D<br/>logvar clamped to [-2, 2]  →  σ_floor = 0.368"]
    
    %% Private sampling — explicit distributions
    H_PRIV --> REPARAMPARAM["z_h ~ N(μ_h, exp(0.5·logvar_h))<br/>logvar clamped to [-4, 2]<br/>σ_floor = 0.135"]
    I_PRIV --> REPARAMPARAM2["z_i ~ N(μ_i, exp(0.5·logvar_i))<br/>logvar clamped to [-4, 2]<br/>σ_floor = 0.135"]
    O_PRIV --> BINCONCRETE["z_o ~ BinaryConcrete(logits, τ)<br/>sigmoid((logits + Logistic(0,1)) / τ)<br/>τ annealed  1.0 → 0.1  over 200 epochs"]
    
    REPARAMPARAM --> Z_H["z_heatmap  32-D"]
    REPARAMPARAM2 --> Z_I["z_impedance  20-D"]
    BINCONCRETE --> Z_O["z_occupancy  32-D"]
    
    %% Shared sampling
    SHARED --> REPARAMSHARED["z_shared ~ N(μ_comb, σ²_comb)<br/>reparameterization trick<br/>z = μ + σ · ε,   ε ~ N(0,1)"]
    REPARAMSHARED --> Z_S["z_shared  48-D"]
    
    %% Final concatenation
    Z_H --> CONCAT["Concatenate All<br/>z_heatmap || z_occupancy ||<br/>z_impedance || z_shared"]
    Z_O --> CONCAT
    Z_I --> CONCAT
    Z_S --> CONCAT
    
    CONCAT --> LATENT["🧬 Full Latent z<br/>132-D<br/>32+32+20+48"]

    %% Styling
    style INPUT fill:#e0e0e0,stroke:#333,stroke-width:3px
    style H_ENC fill:#ffb3ba,stroke:#333,stroke-width:2px
    style O_ENC fill:#baffc9,stroke:#333,stroke-width:2px
    style I_ENC fill:#ffd8a8,stroke:#333,stroke-width:2px
    style H_PRIV fill:#ffe0e0,stroke:#333,stroke-width:2px
    style O_PRIV fill:#e0ffe0,stroke:#333,stroke-width:2px
    style I_PRIV fill:#fff0e0,stroke:#333,stroke-width:2px
    style H_SHARED fill:#ffcccb,stroke:#333,stroke-width:2px
    style O_SHARED fill:#ccffcc,stroke:#333,stroke-width:2px
    style I_SHARED fill:#ffe6cc,stroke:#333,stroke-width:2px
    style POE fill:#ff6b6b,color:#fff,stroke:#333,stroke-width:3px
    style SHARED fill:#ffa94d,stroke:#333,stroke-width:2px
    style REPARAMPARAM fill:#fff9e6,stroke:#333,stroke-width:2px
    style REPARAMPARAM2 fill:#fff9e6,stroke:#333,stroke-width:2px
    style BINCONCRETE fill:#e6f3ff,stroke:#333,stroke-width:2px
    style Z_H fill:#ffe6e6,stroke:#333,stroke-width:2px
    style Z_O fill:#e6ffe6,stroke:#333,stroke-width:2px
    style Z_I fill:#ffe6cc,stroke:#333,stroke-width:2px
    style Z_S fill:#ffa94d,stroke:#333,stroke-width:2px
    style CONCAT fill:#aed9e0,stroke:#333,stroke-width:2px
    style LATENT fill:#b4a7f5,stroke:#333,stroke-width:3px
            </div>

            <div class="info-box">
                <h3>📊 Product of Experts Encoder — Distributions:</h3>
                <ul>
                    <li><strong>🔥 Heatmap private  z_h ~ N(μ_h, σ²_h):</strong> σ_h = exp(0.5·logvar_h), logvar clamped to [−4, 2] → σ_floor = 0.135</li>
                    <li><strong>🗺️ Occupancy private  z_o ~ BinaryConcrete(logits, τ):</strong> z = sigmoid((logits + Logistic(0,1)) / τ); τ annealed from 1.0 → 0.1 over 200 epochs; straight-through gradient in backward pass</li>
                    <li><strong>📊 Impedance private  z_i ~ N(μ_i, σ²_i):</strong> same parameterization as heatmap, logvar clamped to [−4, 2]</li>
                    <li><strong>⚡ Shared PoE posterior  z_s ~ N(μ_comb, σ²_comb):</strong> precision sum: 1/σ²_comb = Σ(1/σ²_i) + 1, weighted mean: μ_comb = σ²_comb · Σ(μ_i/σ²_i); unit Gaussian N(0,1) is the 4th expert (prior); logvar clamped to [−2, 2] → σ_floor = 0.368</li>
                    <li><strong>🔄 Reparameterization trick:</strong> z = μ + σ·ε,  ε ~ N(0,1) — enables gradients to flow through sampling for Gaussian branches</li>
                    <li><strong>🧬 Full latent z  132-D:</strong> [z_h (32) ‖ z_o (32) ‖ z_i (20) ‖ z_s (48)]</li>
                </ul>
            </div>
        </div>

        <div id="decoder" class="tab-content">
            <div class="info-box">
                <h2>🎯 Multi-Modal Decoder with Private+Shared Reconstruction</h2>
                <p>Latent z (132D: 32+32+20+48) → Split into private+shared → 3 independent reconstruction heads</p>
            </div>

            <div class="mermaid">
flowchart TB
    LATENT["🧬 Latent z — 132-D<br/>32 + 32 + 20 + 48"]
    LATENT --> SP["Split into modality components<br/>z_heatmap: 32-D  |  z_occupancy: 32-D<br/>z_impedance: 20-D  |  z_shared: 48-D"]

    SP --> H_IN["Input 80-D<br/>z_heatmap + z_shared"]
    SP --> O_IN["Input 80-D<br/>z_occupancy + z_shared"]
    SP --> I_IN["Input 68-D<br/>z_impedance + z_shared"]

    subgraph HD["Heatmap Decoder"]
        H_IN --> H_FC["FC  80 → 256"]
        H_FC --> H_FC2["FC  256 → 4096"]
        H_FC2 --> H_RESHAPE["Reshape  64 × 8 × 8"]
        H_RESHAPE --> H_C1["ConvTranspose2d  64→32<br/>8×8 → 16×16"]
        H_C1 --> H_C2["ConvTranspose2d  32→16<br/>16×16 → 32×32"]
        H_C2 --> H_C3["ConvTranspose2d  16→1<br/>32×32 → 64×64"]
        H_C3 --> H_OUT["🔥 Heatmap  1×64×64"]
    end

    subgraph OD["Occupancy Decoder"]
        O_IN --> O_F1["FC  80 → 256"]
        O_F1 --> O_F2["FC  256 → 512"]
        O_F2 --> O_F3["FC  512 → 512"]
        O_F3 --> O_F4["FC  512 → 256"]
        O_F4 --> O_F5["FC  256 → 52"]
        O_F5 --> O_OUT["🗺️ Occupancy  52-D<br/>Sigmoid activation"]
    end

    subgraph ID["Impedance Decoder"]
        I_IN --> I_F1["FC  68 → 256"]
        I_F1 --> I_F2["FC  256 → 512"]
        I_F2 --> I_F3["FC  512 → 462"]
        I_F3 --> I_OUT["📊 Impedance  2×231<br/>raw + derivative channels"]
    end

    style LATENT   fill:#b4a7f5,stroke:#333,stroke-width:3px
    style SP       fill:#aed9e0,stroke:#333,stroke-width:2px
    style H_IN     fill:#fff9e6,stroke:#333,stroke-width:2px
    style O_IN     fill:#fff9e6,stroke:#333,stroke-width:2px
    style I_IN     fill:#fff9e6,stroke:#333,stroke-width:2px
    style H_OUT    fill:#ffb3ba,stroke:#333,stroke-width:2px
    style O_OUT    fill:#baffc9,stroke:#333,stroke-width:2px
    style I_OUT    fill:#ffd8a8,stroke:#333,stroke-width:2px
            </div>

            <div class="info-box">
                <h3>📊 Decoder Architecture:</h3>
                <ul>
                    <li><strong>🔥 Heatmap Decoder:</strong> Input 80D (z_heatmap[32D] + z_shared[48D]) → FC (80→256→4096) → Reshape 64×8×8 → 3 ConvTranspose2d layers → 1×64×64 output (z-score normalized)</li>
                    <li><strong>🗺️ Occupancy Decoder:</strong> Input 80D (z_occupancy[32D] + z_shared[48D]) → 5 FC layers (80→256→512→512→256→52) → 52-D binary logits with Sigmoid activation</li>
                    <li><strong>📊 Impedance Decoder (Dual-channel):</strong> Input 68D (z_impedance[20D] + z_shared[48D]) → FC (68→256→512→462) → Reshape 2×231 (raw values + derivatives)</li>
                    <li><strong>🎯 Architecture Philosophy:</strong> Each decoder receives concatenation of [private_latent || shared_latent] to maintain modality-specific structure while leveraging shared cross-modal information</li>
                    <li><strong>✨ Key feature:</strong> All decoders are independent pure FC layers (impedance) or FC+ConvTranspose (heatmap), no shared hidden layer bottleneck</li>
                </ul>
            </div>
        </div>

        <div id="pipeline" class="tab-content">
            <div class="info-box">
                <h2>📊 Data Pipeline</h2>
                <p>End-to-end workflow: normalization, training, and inference.</p>
            </div>

            <div class="mermaid">
flowchart TB
    subgraph LOAD["Data Loading"]
        H_IN["Heatmap<br/>1x64x64"] --> DL["DataLoader<br/>batch = 64"]
        O_IN["Occupancy<br/>52-D binary"]  --> DL
        I_IN["Impedance<br/>2x231"] --> DL
    end

    subgraph TRAIN["Training Loop"]
        ENC["Encoder<br/>3 independent branches"] --> POE["Product of Experts<br/>precision-weighted fusion"]
        POE --> LAT["Latent z — 132-D<br/>32 + 32 + 20 + 48"]
        LAT --> DEC["Decoder<br/>3 reconstruction heads"]
        DEC --> H_P["Heatmap Pred<br/>1x64x64"]
        DEC --> O_P["Occupancy Pred<br/>52-D"]
        DEC --> I_P["Impedance Pred<br/>2x231"]
        H_P --> LH["MSE Loss"]
        O_P --> LO["BCE Loss"]
        I_P --> LI["MAE Loss"]
        LAT --> KL["KL Divergence"]
        LH --> LOSS["Total Loss<br/>L_h + L_o + L_i + beta x KL"]
        LO --> LOSS
        LI --> LOSS
        KL --> LOSS
        LOSS --> OPT["Adam  lr=1e-4  —  300 epochs"]
    end

    subgraph INFER["Inference"]
        ZS["Sample z from N(0,1)"] --> DI["VAE Decoder"]
        DI --> RH["Heatmap<br/>1x64x64"]
        DI --> RO["Occupancy<br/>52-D"]
        DI --> RI["Impedance<br/>2x231"]
    end

    DL --> ENC

    style H_IN fill:#ffb3ba,stroke:#333,stroke-width:2px
    style O_IN fill:#baffc9,stroke:#333,stroke-width:2px
    style I_IN fill:#ffd8a8,stroke:#333,stroke-width:2px
    style DL  fill:#e6f3ff,stroke:#333,stroke-width:2px
    style POE fill:#ff6b6b,color:#fff,stroke:#333,stroke-width:3px
    style LAT fill:#b4a7f5,stroke:#333,stroke-width:3px
    style H_P fill:#ffb3ba,stroke:#333,stroke-width:2px
    style O_P fill:#baffc9,stroke:#333,stroke-width:2px
    style I_P fill:#ffd8a8,stroke:#333,stroke-width:2px
    style LOSS fill:#d4edda,stroke:#333,stroke-width:2px
    style OPT fill:#e6f3ff,stroke:#333,stroke-width:2px
    style ZS  fill:#e0e0e0,stroke:#333,stroke-width:2px
            </div>

            <div class="info-box">
                <h3>📊 Pipeline Details:</h3>
                <ul>
                    <li><strong>🗂️ Input Data:</strong> Heatmap (1×64×64 normalized), Occupancy (52-D binary), Impedance (2×231 dual-channel with derivatives)</li>
                    <li><strong>📦 Data Loading:</strong> Batch size 64, normalized z-score data for all modalities</li>
                    <li><strong>🧬 Encoding:</strong> 3 independent encoders → feature extraction (64D each) → private latent projections + shared space predictions</li>
                    <li><strong>⚡ PoE Fusion:</strong> Each modality predicts shared latent with uncertainty → precision-weighted fusion → 48D shared latent</li>
                    <li><strong>🔄 Decoding:</strong> Split z into [private || shared] for each modality → 3 independent decoders → reconstruction outputs</li>
                    <li><strong>⚖️ Loss:</strong> MSE (heatmap) + BCE with logits (occupancy) + MAE (impedance) + β-annealed KL regularization</li>
                    <li><strong>🎯 Optimization:</strong> Adam optimizer (lr=1e-4), 300 epochs, β annealing from 0.0 to 0.01</li>
                </ul>
            </div>
        </div>

        <div id="losses" class="tab-content">
            <div class="info-box">
                <h2>⚖️ Loss Functions</h2>
                <p>Multi-task loss with specialized losses per modality and KL annealing.</p>
            </div>

            <div class="mermaid">
flowchart TB
    subgraph "Heatmap Loss"
        H_PRED["Heatmap Pred<br/>1×64×64"]
        H_TARGET["Heatmap Target<br/>1×64×64"]
        
        H_PRED --> H_MSE["Mean Squared<br/>Error MSE"]
        H_TARGET --> H_MSE
        H_MSE --> H_LOSS["L_heatmap"]
    end
    
    subgraph "Occupancy Loss"
        O_PRED["Occupancy Logits<br/>52-D"]
        O_TARGET["Occupancy Target<br/>52-D binary"]
        
        O_PRED --> O_BCE["Binary Cross Entropy<br/>With Logits"]
        O_TARGET --> O_BCE
        O_BCE --> O_LOSS["L_occupancy"]
    end
    
    subgraph "Impedance Loss"
        I_PRED["Impedance Pred<br/>2×231"]
        I_TARGET["Impedance Target<br/>2×231"]
        
        I_PRED --> I_MAE["Mean Absolute<br/>Error L1"]
        I_TARGET --> I_MAE
        I_MAE --> I_LOSS["L_impedance<br/>MAE for derivatives"]
    end
    
    subgraph "KL Regularization with Beta Annealing"
        MU["μ<br/>132-D"]
        LOGVAR["logvar<br/>132-D"]
        
        MU --> KL_CALC["KL ~ -0.5 × sum<br/>1 + logvar - μ² - exp(logvar)"]
        LOGVAR --> KL_CALC
        KL_CALC --> KL_RAW["L_KL_raw"]
        
        BETA["beta annealing<br/>0.0 → 0.01<br/>epochs 0-200"]
        KL_RAW --> KL_MULT["Multiply"]
        BETA --> KL_MULT
        KL_MULT --> KL_LOSS["L_KL = β × L_KL_raw"]
    end
    
    subgraph CROSS["Cross-Modal Reconstruction Loss"]
        direction TB
        PARTIAL["Encode subset of modalities<br/>e.g. Heatmap only at inference"]
        PARTIAL --> POE_P["Partial PoE:<br/>observed experts + unit Gaussian prior<br/>missing modalities contribute only 1/σ²_prior = 1"]
        POE_P --> Z_CM["z_shared from partial obs<br/>shared space captures cross-modal info"]
        Z_CM --> DEC_CM["Decode all 3 modalities<br/>from partial encoding"]
        DEC_CM --> L_CM["L_cross = L_h + L_o + L_i<br/>over predicted modalities only"]
    end

    H_LOSS --> TOTAL["Total Loss"]
    O_LOSS --> TOTAL
    I_LOSS --> TOTAL
    KL_LOSS --> TOTAL
    L_CM -.->|optional cross-modal term| TOTAL
    
    TOTAL --> FINAL["L_total = L_h + L_o + L_i + β·L_KL  (+  λ·L_cross)"]
    
    style H_LOSS fill:#ffb3ba
    style O_LOSS fill:#baffc9
    style I_LOSS fill:#ffd8a8
    style KL_LOSS fill:#b4a7f5
    style BETA fill:#ffeaa7
    style TOTAL fill:#d4edda
    style FINAL fill:#d4edda
    style PARTIAL fill:#e6f3ff,stroke:#4a90d9,stroke-width:2px
    style POE_P fill:#ff6b6b,color:#fff,stroke:#333,stroke-width:2px
    style Z_CM fill:#b4a7f5,stroke:#333,stroke-width:2px
    style L_CM fill:#fff0e6,stroke:#e67e22,stroke-width:2px
            </div>

            <div class="info-box">
                <h3>📊 Loss Function Details + Cross-Modal Reconstruction:</h3>
                <ul>
                    <li><strong>🔥 Heatmap Loss:</strong> MSE (reduction='mean') on z-score normalized 1×64×64 output</li>
                    <li><strong>🗺️ Occupancy Loss:</strong> BCEWithLogitsLoss — numerically stable, handles 52-D binary vector</li>
                    <li><strong>📊 Impedance Loss:</strong> MAE / L1 — robust to outliers in dual-channel 2×231 signal</li>
                    <li><strong>🧬 KL Divergence:</strong> −0.5 · Σ(1 + logσ² − μ² − σ²) over all 132 latent dims; β annealed 0.0 → 0.01 over epochs 0–200 to prevent posterior collapse</li>
                    <li><strong>🔀 Cross-Modal Reconstruction:</strong> At inference time, encode only a subset of modalities; missing modalities contribute only their prior (unit Gaussian) to the PoE. The shared z_s still captures the cross-modal structure, so all modalities can be decoded — enabling <em>modality imputation</em> from partial observations</li>
                    <li><strong>⚡ Partial PoE formula:</strong> 1/σ²_comb = 1/σ²_observed + 1  (prior precision = 1); μ_comb = σ²_comb · (μ_obs/σ²_obs + 0) — unobserved modalities simply fall back to the prior</li>
                    <li><strong>🎯 Total training loss:</strong> L_total = L_h + L_o + L_i + β·L_KL  (cross-modal term λ·L_cross is optional at training time)</li>
                </ul>
            </div>
        </div>

        <div id="results" class="tab-content">
            <div class="info-box">
                <h2>🖼️ Generated vs Real — Visual Results</h2>
                <p>Side-by-side comparison of VAE reconstructions against real samples across all three modalities.</p>
            </div>

            <div class="result-item">
                <h3>🔥 Heatmap — Generated vs Real</h3>
                <img src="__HEATMAP_SRC__" alt="Generated vs Real Heatmap" style="max-width:100%;border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,0.12);"/>
            </div>

            <div class="result-item" style="margin-top:30px;">
                <h3>📊 Impedance Profile — Generated vs Real</h3>
                <img src="__IMPEDANCE_SRC__" alt="Generated vs Real Impedance" style="max-width:100%;border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,0.12);"/>
            </div>

            <div class="result-item" style="margin-top:30px;">
                <h3>🗺️ Occupancy Map — Comparison</h3>
                <img src="__OCCUPANCY_SRC__" alt="Occupancy Comparison" style="max-width:100%;border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,0.12);"/>
            </div>
        </div>

        <div class="footer">
            <p>Use the tabs to explore encoder flow, decoder hierarchy, composed loss, or experimental results.</p>
            <p>Right-click any diagram or image to export.</p>
        </div>
    </div>

    <script>
        console.log('Starting Mermaid initialization...');
        
        mermaid.initialize({
            startOnLoad: true,
            theme: 'default',
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis',
                nodeSpacing: 50,
                rankSpacing: 60,
                padding: 15
            },
            arrowMarkerAbsolute: false
        });
        
        console.log('Mermaid initialized');

        // Force render all diagrams by temporarily showing all tabs
        window.addEventListener('DOMContentLoaded', () => {
            console.log('DOM loaded, forcing initial render of all diagrams...');
            
            const allTabs = document.querySelectorAll('.tab-content');
            const originalDisplays = [];
            
            // Temporarily show all tabs
            allTabs.forEach((tab, index) => {
                originalDisplays[index] = tab.style.display;
                tab.style.display = 'block';
            });
            
            // Give Mermaid time to render
            setTimeout(() => {
                // Restore original display states
                allTabs.forEach((tab, index) => {
                    if (!tab.classList.contains('active')) {
                        tab.style.display = 'none';
                    }
                });
                console.log('All diagrams rendered, tabs restored');
            }, 500);
        });

        function showTab(evt, tabName) {
            console.log('Switching to tab:', tabName);
            
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => {
                content.classList.remove('active');
                content.style.display = 'none';
            });

            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));

            const targetTab = document.getElementById(tabName);
            targetTab.classList.add('active');
            targetTab.style.display = 'block';
            evt.target.classList.add('active');
            
            console.log('Tab switched to:', tabName);
        }
    </script>
</body>
</html>
"""


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(output_dir, 'vae_architecture_diagram.html')

    workspace = os.path.dirname(output_dir)
    heatmap_src   = encode_image(os.path.join(workspace, 'temp_visuals', 'generated_vs_real_heatmap.png'))
    impedance_src = encode_image(os.path.join(workspace, 'temp_visuals', 'generated_vs_real_impedance_profile.png'))
    occupancy_src = encode_image(os.path.join(workspace, 'temp_visuals', 'occupancy_comparison.png'))

    html = HTML_TEMPLATE.replace('__HEATMAP_SRC__',   heatmap_src)
    html = html.replace('__IMPEDANCE_SRC__', impedance_src)
    html = html.replace('__OCCUPANCY_SRC__', occupancy_src)

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    size_mb = len(html.encode('utf-8')) / (1024 * 1024)
    print(f"HTML created: {html_path}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
