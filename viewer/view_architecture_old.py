#!/usr/bin/env python3
"""
Single script to generate and view VAE architecture diagrams in a browser.
Usage: python3 view_architecture.py
"""

import os
import time
import subprocess
import webbrowser
import signal
import sys
import base64
from pathlib import Path

def encode_image(img_path):
    """Encode image to base64 data URI"""
    try:
        with open(img_path, 'rb') as f:
            img_data = f.read()
            encoded = base64.b64encode(img_data).decode('utf-8')
            return f"data:image/png;base64,{encoded}"
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
        .result-images {
            display: flex;
            flex-direction: column;
            gap: 30px;
        }
        .result-item {
            background: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .result-item h3 {
            color: #2c3e50;
            margin-top: 0;
            margin-bottom: 15px;
        }
        .result-item img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
        }
        .occupancy-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        .occupancy-item {
            text-align: center;
        }
        .occupancy-item h4 {
            color: #34495e;
            margin-bottom: 10px;
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
        </div>

        <div id="encoder" class="tab-content active">
            <div class="info-box">
                <h2>🧩 Product of Experts (PoE) Multi-Modal Encoder</h2>
                <p>Four branches (heatmap, occupancy, impedance, max_value) encode independently. Each modality predicts the shared latent space with uncertainty, combined via precision-weighted Product of Experts fusion.</p>
            </div>

            <div class="mermaid">
graph TB
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
    H_SHARED --> POE["⚡ Product of Experts<br/>Combine 3 Experts + Prior<br/>1/σ²_comb = Σ1/σ²_i<br/>μ_comb = σ²_comb·Σμ_i/σ²_i"]
    O_SHARED --> POE
    I_SHARED --> POE
    
    POE --> SHARED["Shared Space<br/>μ_s, logvar_s<br/>48-D"]
    
    %% Private sampling with occupancy
    H_PRIV --> REPARAMPARAM["H: Reparameterize<br/>z_h ~ N(μ_h,σ²_h)"]
    I_PRIV --> REPARAMPARAM2["I: Reparameterize<br/>z_i ~ N(μ_i,σ²_i)"]
    O_PRIV --> BINCONCRETE["O: Binary Concrete<br/>Sample from logits"]
    
    REPARAMPARAM --> Z_H["z_heatmap<br/>32-D"]
    REPARAMPARAM2 --> Z_I["z_impedance<br/>20-D"]
    BINCONCRETE --> Z_O["z_occupancy<br/>32-D"]
    
    %% Shared sampling
    SHARED --> REPARAMSHARED["Reparameterize<br/>z_shared ~ N(μ_s,σ²_s)"]
    REPARAMSHARED --> Z_S["z_shared<br/>48-D"]
    
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
                <h3>📊 Product of Experts Encoder Architecture:</h3>
                <ul>
                    <li><strong>🔥 Heatmap Branch:</strong> Input 1×64×64 → Conv2d (1→16→32→64) → FC (4096→512→64D) features</li>
                    <li><strong>👥 Private Heatmap:</strong> 64D features → μ_h, logvar_h → 32D (Gaussian)</li>
                    <li><strong>🗺️ Occupancy Branch:</strong> Input 52-D vector → Linear (52→128→256→128→64D) features</li>
                    <li><strong>👥 Private Occupancy:</strong> 64D features → logits → 32D (Binary Concrete sampling with temperature annealing)</li>
                    <li><strong>📊 Impedance Branch (Dual-channel):</strong> Input 2×231 (raw + derivatives) → Conv1d (2→16→32→64→128) → FC (1920→256→64D) features</li>
                    <li><strong>👥 Private Impedance:</strong> 64D features → μ_i, logvar_i → 20D (Gaussian)</li>
                    <li><strong>⚡ Shared Space (PoE):</strong> Each modality predicts μ, logvar (48D each) via separate FC layers → Combined via precision-weighted Product of Experts</li>
                    <li><strong>📐 PoE Formula:</strong> Precision sum: $1/σ²_{comb} = \sum(1/σ²_i)$, Mean: $μ_{comb} = σ²_{comb} \cdot \sum(μ_i/σ²_i)$</li>
                    <li><strong>🧬 Total Latent:</strong> 132D = 32D (heatmap) + 32D (occupancy) + 20D (impedance) + 48D (shared PoE)</li>
                    <li><strong>✨ Benefits:</strong> Uncertainty-aware fusion, handles heterogeneous modalities, automatic confidence weighting via precision</li>
                </ul>
            </div>
        </div>

        <div id="decoder" class="tab-content">
            <div class="info-box">
                <h2>🎯 Multi-Modal Decoder with Private+Shared Reconstruction</h2>
                <p>Latent z (132D: 32+32+20+48) → Split into private+shared → 3 independent reconstruction heads</p>
            </div>

            <div class="mermaid">
graph TB
    LATENT["🧬 Latent z<br/>132-D<br/>32+32+20+48"]
    
    LATENT --> LATENT_SPLIT["Split Latent<br/>z_heatmap: 32-D<br/>z_occupancy: 32-D<br/>z_impedance: 20-D<br/>z_shared: 48-D"]
    
    LATENT_SPLIT --> CONCAT_DEC["Prepare Decoder Inputs<br/>[z_priv | z_shared]"]
    
    subgraph "Heatmap Decoder (z_hm + z_shared = 80D)"
        CONCAT_DEC --> H_FC[FC 80→256]
        H_FC --> H_FC2[FC 256→4096]
        H_FC2 --> H_RESHAPE[Reshape 64×8×8]
        H_RESHAPE --> H_CONV1[ConvTranspose2d<br/>64→32, 8×8→16×16]
        H_CONV1 --> H_CONV2[ConvTranspose2d<br/>32→16, 16×16→32×32]
        H_CONV2 --> H_CONV3[ConvTranspose2d<br/>16→1, 32×32→64×64]
        H_CONV3 --> H_OUT["🔥 Heatmap Reconstruction<br/>1×64×64<br/>z-score normalized"]  
    end
    
    subgraph "Occupancy Decoder (z_occ + z_shared = 80D)"
        CONCAT_DEC --> O_FC1[FC 80→256]
        O_FC1 --> O_FC2[FC 256→512]
        O_FC2 --> O_FC3[FC 512→512]
        O_FC3 --> O_FC4[FC 512→256]
        O_FC4 --> O_FC5[FC 256→52]
        O_FC5 --> O_OUT["🗺️ Occupancy Logits<br/>52-D<br/>Binary, Sigmoid activation"]
    end
    
    subgraph "Impedance Decoder (z_imp + z_shared = 68D)"
        CONCAT_DEC --> I_FC1[FC 68→256]
        I_FC1 --> I_FC2[FC 256→512]
        I_FC2 --> I_FC3[FC 512→462]
        I_FC3 --> I_OUT["📊 Impedance Reconstruction<br/>2×231<br/>raw + derivative channels<br/>z-score normalized"]
    end
    
    style LATENT fill:#b4a7f5,stroke:#333,stroke-width:3px
    style CONCAT_DEC fill:#aed9e0,stroke:#333,stroke-width:2px
    style H_OUT fill:#ffb3ba,stroke:#333,stroke-width:2px
    style O_OUT fill:#baffc9,stroke:#333,stroke-width:2px
    style I_OUT fill:#ffd8a8,stroke:#333,stroke-width:2px
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
graph TB
    subgraph "� Dataset Statistics"
        DS[15K Samples<br/>Quality: 8/8 EXCELLENT]
        DS --> H_STATS[Heatmap: CV=2.81<br/>exceptional diversity]
        DS --> I_STATS[Impedance: CV=0.38<br/>high diversity<br/>skew 2.72→-0.10]
        DS --> O_STATS[Occupancy: CV=0.31<br/>100% unique patterns<br/>46.4% density]
    end
    
    subgraph "🗂️ Data Loading"
        HEATMAP[Load Heatmap<br/>64x64x2]
        OCCUPANCY[Load Occupancy<br/>7x8x1]
        IMPEDANCE[Load Impedance<br/>231x1 normalized]
        MAXVALUE[Load MaxValue<br/>scalar]
        
        HEATMAP --> DATALOADER[DataLoader<br/>batch_size=64]
        OCCUPANCY --> DATALOADER
        IMPEDANCE --> DATALOADER
        MAXVALUE --> DATALOADER
    end
    
    subgraph "🏋️ Training Pipeline with PoE"
        DATALOADER --> ENCODER[VAE Encoder<br/>4 branches]
        ENCODER --> POE[Product of Experts<br/>precision-weighted fusion]
        POE --> LATENT[Latent z<br/>128D<br/>64 priv + 64 shared]
        
        LATENT --> DECODER[VAE Decoder<br/>4 heads]
        
        DECODER --> H_PRED[Heatmap Pred 64x64x2]
        DECODER --> O_PRED[Occupancy Pred 7x8x1]
        DECODER --> I_PRED[Impedance Pred 231x1]
        DECODER --> M_PRED[MaxValue Pred scalar]
        
        H_PRED --> L_H[MSE Loss]
        HEATMAP --> L_H
        
        O_PRED --> L_O[BCE Loss]
        OCCUPANCY --> L_O
        
        I_PRED --> L_I[MAE Loss]
        IMPEDANCE --> L_I
        
        M_PRED --> L_M[MSE Loss]
        MAXVALUE --> L_M
        
        LATENT --> KL_DIV[KL Divergence]
        
        L_H --> TOTAL[Total Loss]
        L_O --> TOTAL
        L_I --> TOTAL
        L_M --> TOTAL
        KL_DIV --> TOTAL
        
        TOTAL --> BACKPROP[Backprop<br/>Adam lr=1e-5]
    end
    
    subgraph "🔮 Inference"
        Z_SAMPLE[Sample z ~ N(0,1)]
        Z_SAMPLE --> DEC_INF[VAE Decoder]
        DEC_INF --> H_OUT[Heatmap 64x64x2]
        DEC_INF --> O_OUT[Occupancy 7x8x1]
        DEC_INF --> I_OUT[Impedance 231x1]
        DEC_INF --> M_OUT[MaxValue scalar]
    end
    
    style DS fill:#e6f3ff
    style POE fill:#ff6b6b,color:#fff
    style LATENT fill:#b4a7f5
    style H_PRED fill:#ffb3ba
    style O_PRED fill:#baffc9
    style I_PRED fill:#ffd8a8
    style M_PRED fill:#ffaaee
    style TOTAL fill:#d4edda
            </div>

            <div class="info-box">
                <h3>📊 Pipeline Details:</h3>
                <ul>
                    <li><strong>🗂️ Input Data:</strong> Heatmap (1×64×64 normalized), Occupancy (52-D binary), Impedance (2×231 dual-channel with derivatives)</li>
                    <li><strong>📦 Data Loading:</strong> Batch size 64, 4 workers, normalized z-score data for all modalities</li>
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
                <p>Multi-task loss with uncertainty-based automatic weighting, specialized losses for class imbalance, and KL annealing.</p>
            </div>

            <div class="mermaid">
graph TB
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
        
        MU --> KL_CALC["KL ∼ -0.5 × sum<br/>1 + logvar - μ² - exp(logvar)"]
        LOGVAR --> KL_CALC
        KL_CALC --> KL_RAW["L_KL_raw"]
        
        BETA["beta annealing<br/>0.0 → 0.01<br/>epochs 0-200"]
        KL_RAW --> KL_MULT["Multiply"]
        BETA --> KL_MULT
        KL_MULT --> KL_LOSS["L_KL = β × L_KL_raw"]
    end
    
    H_LOSS --> TOTAL["Total Loss"]
    O_LOSS --> TOTAL
    I_LOSS --> TOTAL
    KL_LOSS --> TOTAL
    
    TOTAL --> FINAL["L_total = L_h + L_o + L_i + β·L_KL"]
    
    style H_LOSS fill:#ffb3ba
    style O_LOSS fill:#baffc9
    style I_LOSS fill:#ffd8a8
    style KL_LOSS fill:#b4a7f5
    style BETA fill:#ffeaa7
    style TOTAL fill:#d4edda
    style FINAL fill:#d4edda
            </div>

            <div class="info-box">
                <h3>📊 Loss Function Details:</h3>
                <ul>
                    <li><strong>🔥 Heatmap Loss:</strong> MSE (reduction='mean') on normalized heatmap 1×64×64, z-score normalized</li>
                    <li><strong>🗺️ Occupancy Loss:</strong> Binary Cross Entropy with Logits (BCEWithLogitsLoss) for 52-D binary vector classification</li>
                    <li><strong>📊 Impedance Loss:</strong> MAE/L1 Loss for dual-channel (2×231) impedance prediction. Better for skewed data with outliers</li>
                    <li><strong>🧬 KL Divergence:</strong> Regularizes latent z (132D) to N(0,1). Formula: $-0.5 \times \sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$</li>
                    <li><strong>⏳ Beta Annealing:</strong> β goes from 0.0 → 0.01 over epochs 0-200 to prevent posterior collapse and enable learning</li>
                    <li><strong>🔧 Clamping:</strong> Private logvars clamped to [-4, 2] (σ floor=0.135), shared logvar clamped to [-2, 2] (σ floor=0.368)</li>
                    <li><strong>⚖️ Weighting:</strong> All reconstruction losses use reduction='mean' for comparable scales</li>
                    <li><strong>🎯 Total:</strong> $L_{total} = L_h + L_o + L_i + \beta \cdot L_{KL}$, batch size 64, lr=1e-4, 300 epochs</li>
                </ul>
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
                htmlLabels: true
            }
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
    print("🔄 Encoding images into HTML...")
    os.makedirs('temp_visuals', exist_ok=True)
    html_path = 'temp_visuals/vae_architecture_diagram.html'
    
    # Encode all images
    heatmap_src = encode_image("experiments/exp020/visuals_2/generated_vs_real_heatmap.png")
    impedance_src = encode_image("experiments/exp020/visuals_2/generated_vs_real_impedance_profile.png")
    occ0_src = encode_image("experiments/exp012/visuals/data_sample_0/occupancy_map_visual.png")
    occ1_src = encode_image("experiments/exp012/visuals/data_sample_1/occupancy_map_visual.png")
    occ2_src = encode_image("experiments/exp012/visuals/data_sample_2/occupancy_map_visual.png")
    occ3_src = encode_image("experiments/exp012/visuals/data_sample_3/occupancy_map_visual.png")
    occ4_src = encode_image("experiments/exp012/visuals/data_sample_4/occupancy_map_visual.png")
    
    # Replace placeholders with actual base64 data
    html_content = HTML_TEMPLATE.replace('__HEATMAP_SRC__', heatmap_src)
    html_content = html_content.replace('__IMPEDANCE_SRC__', impedance_src)
    html_content = html_content.replace('__OCC0_SRC__', occ0_src)
    html_content = html_content.replace('__OCC1_SRC__', occ1_src)
    html_content = html_content.replace('__OCC2_SRC__', occ2_src)
    html_content = html_content.replace('__OCC3_SRC__', occ3_src)
    html_content = html_content.replace('__OCC4_SRC__', occ4_src)
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("✅ Architecture diagram HTML created with embedded images")
    print(f"   File size: {len(html_content) / (1024*1024):.2f} MB")

    port = 8899
    print(f"🌐 Starting HTTP server on port {port}...")

    # Run server from temp_visuals since images are embedded
    server_process = subprocess.Popen(
        ['python3', '-m', 'http.server', str(port)],
        cwd='temp_visuals',
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(1)

    url = f'http://localhost:{port}/vae_architecture_diagram.html'
    print(f"🚀 Opening diagram in browser: {url}")
    print("📋 Press Ctrl+C to stop the server")

    try:
        webbrowser.open(url)
    except Exception:
        print(f"   Manually open: {url}")

    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        server_process.terminate()
        server_process.wait()
        print("✅ Server stopped")

if __name__ == '__main__':
    main()
