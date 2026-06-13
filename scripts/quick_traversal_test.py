"""
Quick Latent Traversal Example
============================

A simplified script to quickly test latent traversal on a few key dimensions.
Use this to verify the setup before running the full analysis.

Usage:
    python quick_traversal_test.py
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from source.model.vae_multi_input import MultiInputVAE

# Quick test configuration
CHECKPOINT_PATH = "experiments/exp012/checkpoints/epoch_100.pt"
LATENT_DIM = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("temp_visuals/quick_traversal_test")

def quick_traversal_test():
    """Test latent traversal on a single dimension"""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = MultiInputVAE(latent_dim=LATENT_DIM)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Test traversal on dimension 0
    dim_to_test = 0
    traversal_values = np.linspace(-2, 2, 5)  # Just 5 points for quick test
    
    print(f"Testing traversal on dimension {dim_to_test}")
    
    results = []
    base_latent = torch.zeros(1, LATENT_DIM, device=DEVICE)
    
    with torch.no_grad():
        for i, value in enumerate(traversal_values):
            print(f"  Generating sample {i+1}/5 (z[{dim_to_test}] = {value:.1f})")
            
            z = base_latent.clone()
            z[0, dim_to_test] = value
            
            heatmap, occupancy_logits, impedance = model.decode(z)
            occupancy = torch.sigmoid(occupancy_logits)
            
            results.append({
                'value': value,
                'heatmap': heatmap.cpu().numpy()[0],
                'occupancy': occupancy.cpu().numpy()[0],
                'impedance': impedance.cpu().numpy()[0]
            })
    
    # Create quick visualization
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    for i, result in enumerate(results):
        value = result['value']
        
        # Heatmap (channel 0)
        ax = axes[0, i]
        im = ax.imshow(result['heatmap'][0], cmap='jet', vmin=0, vmax=1)
        ax.set_title(f'Heatmap\\nz[{dim_to_test}]={value:.1f}')
        ax.axis('off')
        
        # Occupancy
        ax = axes[1, i]
        occ_binary = (result['occupancy'][0] > 0.5).astype(float)
        ax.imshow(occ_binary, cmap='Greys', vmin=0, vmax=1)
        ax.set_title(f'Occupancy\\nz[{dim_to_test}]={value:.1f}')
        ax.axis('off')
        
        # Impedance (just show the curve shape)
        ax = axes[2, i] 
        ax.plot(result['impedance'], 'b-')
        ax.set_title(f'Impedance\\nz[{dim_to_test}]={value:.1f}')
        ax.set_ylim(0, 1)  # Normalized impedance range
    
    plt.suptitle(f'Quick Traversal Test - Dimension {dim_to_test}', fontsize=16)
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / 'quick_test.png' 
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\\n✓ Quick test completed!")
    print(f"✓ Visualization saved: {save_path}")
    print("\\nIf this works correctly, you can run the full latent_traversal.py script.")
    
    # Print some basic stats
    heatmap_changes = [np.std(r['heatmap']) for r in results]
    occupancy_changes = [np.std(r['occupancy']) for r in results] 
    impedance_changes = [np.std(r['impedance']) for r in results]
    
    print(f"\\nQuick sensitivity analysis for dimension {dim_to_test}:")
    print(f"  Heatmap variation: {np.std(heatmap_changes):.6f}")
    print(f"  Occupancy variation: {np.std(occupancy_changes):.6f}")
    print(f"  Impedance variation: {np.std(impedance_changes):.6f}")

if __name__ == "__main__":
    quick_traversal_test()