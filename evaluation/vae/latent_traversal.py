"""
Latent Traversal Analysis for Multi-Input VAE
=====================================

This script performs systematic latent space exploration to understand which latent 
dimensions control which modalities (heatmap, occupancy, impedance).

Key Analysis:
- Sweeps selected latent dimensions from -2 to +2
- Generates samples at each point
- Visualizes how each modality responds to changes
- Identifies which latent neurons are most influential for each output

Usage:
    python latent_traversal.py
    
Output:
    - Individual traversal plots for each latent dimension
    - Response analysis showing modality sensitivity
    - Combined dashboard showing all traversals
"""
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
import seaborn as sns
from tqdm import tqdm

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[2]
VISUALIZATION_DIR = REPO_ROOT / "evaluation" / "visualization"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VISUALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZATION_DIR))

# Import model and visualization functions
from experiments.exp028_eval_improvement.codes.vae_multi_input_simple import MultiInputVAE
from visualization.simple_visuals import plot_heatmap_array, Impedance_profile, denormalize_impedance
from visualization.occupancy_visual import plot_occupancy_map, build_occupancy_grid_dict


# ============================================================
# CONFIGURATION
# ============================================================

# Model configuration
CHECKPOINT_PATH = "experiments/exp028_eval_improvement/checkpoints/checkpoint_epoch_400.pt"
LATENT_DIM = 32  # Must match training configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Traversal configuration
TRAVERSAL_RANGE = (-2.0, 2.0)  # Range to sweep each latent dimension
NUM_STEPS = 11  # Number of steps in traversal (11 gives: -2, -1.6, -1.2, ..., 1.6, 2.0)
DIMENSIONS_TO_ANALYZE = [0, 1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 31]  # Selected latent dims to analyze

# Conditioning configuration
# The MultiInputVAE used in exp028 is conditioned on K (number of occupied cells).
K_VALUE = 5

# Output configuration  
OUTPUT_DIR = Path("evaluation/latent_traversal")
STATS_PATH = Path("datasets/data_norm/normalization_stats.json")

class LatentTraversalAnalyzer:
    """Analyzes latent space by systematically varying latent dimensions"""
    
    def __init__(self, checkpoint_path: str, latent_dim: int = 32, device: Optional[torch.device] = None):
        """Initialize the analyzer with trained VAE model"""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model = MultiInputVAE(latent_dim=latent_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load normalization stats for impedance denormalization
        self.stats = self._load_stats()
        
        print(f"Model loaded successfully (epoch: {checkpoint.get('epoch', 'unknown')})")
        print(f"Device: {self.device}")
        
    def _load_stats(self) -> dict:
        """Load normalization statistics for denormalization"""
        try:
            with open(STATS_PATH, 'r') as f:
                stats = json.load(f)
            return stats
        except FileNotFoundError:
            print(f"Warning: Stats file not found at {STATS_PATH}")
            print("Will use raw impedance values without denormalization")
            return {}
        except Exception as e:
            print(f"Warning: Error loading stats file: {e}")
            return {}

    def _impedance_mean_std(self) -> Tuple[Optional[float], Optional[float]]:
        """Return (log_mean, log_std) for impedance if available."""
        try:
            imp = self.stats.get("Impedance", {}) if isinstance(self.stats, dict) else {}
            return float(imp["log_mean"]), float(imp["log_std"])
        except Exception:
            return None, None

    def _blend_impedance_channels(self, imp_norm: torch.Tensor) -> torch.Tensor:
        """Blend/integrate impedance channels to a single curve.

        Input: (B,3,231) (or (B,>=2,231))
        Output: (B,231) in the same (normalized) space as channel 0.
        """
        if not (imp_norm.dim() == 3 and imp_norm.shape[1] >= 2):
            if imp_norm.dim() == 3 and imp_norm.shape[1] == 1:
                return imp_norm[:, 0, :]
            return imp_norm

        z_raw = imp_norm[:, 0, :]
        d1 = imp_norm[:, 1, :]
        z_integ1 = torch.cat(
            [z_raw[:, :1], z_raw[:, :1] + torch.cumsum(d1, dim=-1)[:, :-1]],
            dim=-1,
        )

        if imp_norm.shape[1] >= 3:
            d2 = imp_norm[:, 2, :]
            peak_w = torch.abs(d2) / (torch.abs(d2).amax(dim=-1, keepdim=True) + 1e-8)
            pad = 2
            peak_w = torch.nn.functional.pad(peak_w, (pad, pad), mode='reflect') \
                .unfold(-1, 2 * pad + 1, 1).mean(dim=-1)
            z_blend = peak_w * z_raw + (1.0 - peak_w) * z_integ1
        else:
            z_blend = 0.5 * (z_raw + z_integ1)

        return z_blend
    
    def _load_config_files(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load frequency and target impedance data"""
        try:
            frequency = np.load('configs/Frequency_data_hz.npy').squeeze()
            target_impedance = np.load('configs/target_impedance.npy').squeeze()
            return frequency, target_impedance
        except FileNotFoundError as e:
            print(f"Warning: Config file not found: {e}")
            print("Impedance plots will be simplified")
            return None, None
    
    def traverse_dimension(self, dim_idx: int, base_latent: Optional[torch.Tensor] = None,
                          traversal_range: Tuple[float, float] = TRAVERSAL_RANGE,
                          num_steps: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Traverse a single latent dimension while keeping others fixed
        
        Args:
            dim_idx: Index of latent dimension to vary
            base_latent: Base latent vector (if None, uses zeros)
            traversal_range: (min, max) values for traversal
            num_steps: Number of steps in traversal
        
        Returns:
            Dictionary containing arrays of generated samples
        """
        if base_latent is None:
            base_latent = torch.zeros(1, self.latent_dim, device=self.device)
        
        # Create traversal values
        if num_steps is None:
            num_steps = NUM_STEPS
        values = np.linspace(traversal_range[0], traversal_range[1], num_steps)
        
        results = {
            'heatmaps': [],
            'occupancies': [],
            'occupancy_logits': [],
            'impedances': [],       # (231,) Ohm scale (best-effort)
            'impedances_log': [],   # (231,) log(Ohm) (best-effort)
            'impedances_tri': [],   # (3,231) raw model output
            'values': values
        }
        
        with torch.no_grad():
            for value in tqdm(values, desc=f"Traversing dim {dim_idx}"):
                # Create latent vector with modified dimension
                z = base_latent.clone()
                z[0, dim_idx] = value

                K = torch.full((z.shape[0],), K_VALUE, dtype=torch.long, device=self.device)
                
                # Generate sample
                heatmap, occupancy_logits, impedance = self.model.decode(z, K)
                occupancy = torch.sigmoid(occupancy_logits)

                # Impedance output is (B, 3, 231). Blend and denormalize for plotting.
                imp_blend_norm = self._blend_impedance_channels(impedance)
                imp_mean, imp_std = self._impedance_mean_std()
                if imp_mean is not None and imp_std is not None:
                    imp_log = imp_blend_norm * imp_std + imp_mean
                    imp_ohm = torch.exp(imp_log)
                else:
                    imp_log = imp_blend_norm
                    imp_ohm = imp_blend_norm
                
                # Store results
                results['heatmaps'].append(heatmap.cpu().numpy()[0])
                results['occupancies'].append(occupancy.cpu().numpy()[0])
                results['occupancy_logits'].append(occupancy_logits.cpu().numpy()[0])
                results['impedances'].append(imp_ohm.cpu().numpy()[0])
                results['impedances_log'].append(imp_log.cpu().numpy()[0])
                results['impedances_tri'].append(impedance.cpu().numpy()[0])
        
        # Convert to numpy arrays
        results['heatmaps'] = np.array(results['heatmaps'])
        results['occupancies'] = np.array(results['occupancies'])
        results['occupancy_logits'] = np.array(results['occupancy_logits'])
        results['impedances'] = np.array(results['impedances'])
        results['impedances_log'] = np.array(results['impedances_log'])
        results['impedances_tri'] = np.array(results['impedances_tri'])
        
        return results
    
    def analyze_modality_sensitivity(self, results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze how much each modality changes during traversal
        
        Args:
            results: Results from traverse_dimension()
            
        Returns:
            Dictionary with sensitivity scores for each modality
        """
        heatmaps = results['heatmaps']  # (num_steps, C, 64, 64)
        occupancies = results['occupancies']  # (num_steps, 52) probs
        occupancy_logits = results.get('occupancy_logits', None)  # (num_steps, 52) logits
        impedances = results['impedances_log']  # (num_steps, 231)
        
        # Calculate variance across traversal for each modality
        heatmap_var = np.var(heatmaps, axis=0).mean()  # Average variance across spatial dims
        if occupancy_logits is not None:
            occupancy_var = np.var(occupancy_logits, axis=0).mean()
            occupancy_range = (occupancy_logits.max() - occupancy_logits.min())
        else:
            occupancy_var = np.var(occupancies, axis=0).mean()
            occupancy_range = (occupancies.max() - occupancies.min())
        impedance_var = np.var(impedances, axis=0).mean()
        
        # Also calculate range (max - min) as another sensitivity measure
        heatmap_range = (heatmaps.max() - heatmaps.min())
        impedance_range = (impedances.max() - impedances.min())

        # Combinatorial signal: how often the top-K set changes along the traversal.
        topk_flip_rate = None
        try:
            if occupancies.ndim == 2 and occupancies.shape[1] > 0 and K_VALUE > 0:
                topk_sets = []
                k_eff = min(int(K_VALUE), occupancies.shape[1])
                for t in range(occupancies.shape[0]):
                    idx = np.argpartition(-occupancies[t], k_eff - 1)[:k_eff]
                    topk_sets.append(set(map(int, idx.tolist())))
                base = topk_sets[0]
                changes = [len(s.symmetric_difference(base)) / (2.0 * k_eff) for s in topk_sets]
                topk_flip_rate = float(np.mean(changes))
        except Exception:
            topk_flip_rate = None
        
        return {
            'heatmap_variance': float(heatmap_var),
            'occupancy_variance': float(occupancy_var),
            'impedance_variance': float(impedance_var),
            'heatmap_range': float(heatmap_range),
            'occupancy_range': float(occupancy_range),
            'impedance_range': float(impedance_range),
            'occupancy_topk_flip_rate': topk_flip_rate, # type: ignore
        }
    
    def visualize_traversal(self, dim_idx: int, results: Dict[str, np.ndarray], 
                          sensitivity: Dict[str, float]) -> None:
        """Create visualization for a single dimension traversal"""
        
        values = results['values']
        heatmaps = results['heatmaps']
        occupancies = results['occupancies']
        impedances = results['impedances']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Plot a few key traversal points
        key_indices = [0, len(values)//4, len(values)//2, 3*len(values)//4, -1]
        key_values = [values[i] for i in key_indices]
        
        # Top row: Heatmaps (channel 0)
        for i, (idx, val) in enumerate(zip(key_indices, key_values)):
            ax = plt.subplot(4, 5, i+1)
            hm_step = heatmaps[idx]
            hm = hm_step[0] if hm_step.ndim == 3 else hm_step
            im = ax.imshow(hm, cmap='jet', vmin=0, vmax=1, interpolation='bilinear')
            ax.set_title(f'Heatmap\\nz[{dim_idx}]={val:.1f}', fontsize=10)
            ax.axis('off')
            if i == 4:
                plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Second row: Occupancy maps
        for i, (idx, val) in enumerate(zip(key_indices, key_values)):
            ax = plt.subplot(4, 5, i+6)
            occ_vec = occupancies[idx]
            # Show combinatorial decision (top-K) rather than prob>0.5 threshold.
            occ_topk = self._topk_binary(occ_vec, K_VALUE)
            occ_grid = self._occupancy_vec_to_grid(occ_topk)
            occ_binary = (occ_grid > 0.5).astype(float)
            im = ax.imshow(occ_binary, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(f'Occupancy\\nz[{dim_idx}]={val:.1f}', fontsize=10)
            ax.axis('off')
            if i == 4:
                plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Third row: Impedance profiles  
        frequency, target_impedance = self._load_config_files()
        
        for i, (idx, val) in enumerate(zip(key_indices, key_values)):
            ax = plt.subplot(4, 5, i+11)
            
            imp = impedances[idx]
            
            if frequency is not None and target_impedance is not None:
                # Full impedance plot with frequency axis
                ax.loglog(frequency, target_impedance, '--', color='red', alpha=0.5, 
                         label='Target' if i == 0 else '')
                ax.loglog(frequency, imp, '-', color='blue', 
                         label='Generated' if i == 0 else '')
                ax.set_ylim(1e-3, 1e2)
                ax.set_xlabel('Frequency (Hz)' if i == 2 else '')
                ax.set_ylabel('Impedance (Ω)' if i == 0 else '')
            else:
                # Simplified plot without frequency axis
                ax.plot(imp, '-', color='blue')
                ax.set_ylabel('Impedance' if i == 0 else '')
                ax.set_xlabel('Frequency Index' if i == 2 else '')
                
            ax.set_title(f'Impedance\\nz[{dim_idx}]={val:.1f}', fontsize=10)
            ax.grid(True, alpha=0.3)
            if i == 0 and frequency is not None:
                ax.legend(fontsize=8)
        
        # Fourth row: Sensitivity analysis
        ax = plt.subplot(4, 2, 7)
        modalities = ['Heatmap', 'Occupancy', 'Impedance']
        variances = [sensitivity['heatmap_variance'], 
                    sensitivity['occupancy_variance'], 
                    sensitivity['impedance_variance']]
        
        bars = ax.bar(modalities, variances, color=['orange', 'purple', 'brown'])
        ax.set_title(f'Modality Sensitivity (Variance)\\nLatent Dimension {dim_idx}')
        ax.set_ylabel('Variance')
        
        # Add variance values on bars
        for bar, var in zip(bars, variances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{var:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax = plt.subplot(4, 2, 8)
        ranges = [sensitivity['heatmap_range'],
                 sensitivity['occupancy_range'], 
                 sensitivity['impedance_range']]
        
        bars = ax.bar(modalities, ranges, color=['orange', 'purple', 'brown'])
        ax.set_title(f'Modality Sensitivity (Range)\\nLatent Dimension {dim_idx}')
        ax.set_ylabel('Range (Max - Min)')
        
        # Add range values on bars
        for bar, rng in zip(bars, ranges):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rng:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(f'Latent Dimension {dim_idx} Traversal Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f'traversal_dim_{dim_idx:02d}.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Saved traversal visualization: {save_path}")
    
    def run_full_analysis(self, dimensions: List[int] = DIMENSIONS_TO_ANALYZE) -> Dict:
        """
        Run complete latent traversal analysis
        
        Args:
            dimensions: List of latent dimensions to analyze
            
        Returns:
            Complete analysis results
        """
        print(f"Running latent traversal analysis on {len(dimensions)} dimensions...")
        print(f"Dimensions to analyze: {dimensions}")
        
        all_results = {}
        sensitivity_summary = {
            'dimension': [],
            'heatmap_variance': [],
            'occupancy_variance': [],
            'impedance_variance': [],
            'heatmap_range': [],
            'occupancy_range': [],
            'impedance_range': [],
            'occupancy_topk_flip_rate': [],
        }
        
        # Analyze each dimension
        for dim_idx in dimensions:
            print(f"\\n{'='*50}")
            print(f"Analyzing latent dimension {dim_idx}")
            print(f"{'='*50}")
            
            # Perform traversal
            results = self.traverse_dimension(dim_idx)
            
            # Analyze sensitivity
            sensitivity = self.analyze_modality_sensitivity(results)
            
            # Create visualization
            self.visualize_traversal(dim_idx, results, sensitivity)
            
            # Store results
            all_results[dim_idx] = {
                'results': results,
                'sensitivity': sensitivity
            }
            
            # Update summary
            sensitivity_summary['dimension'].append(dim_idx)
            sensitivity_summary['heatmap_variance'].append(sensitivity['heatmap_variance'])
            sensitivity_summary['occupancy_variance'].append(sensitivity['occupancy_variance'])
            sensitivity_summary['impedance_variance'].append(sensitivity['impedance_variance'])
            sensitivity_summary['heatmap_range'].append(sensitivity['heatmap_range'])
            sensitivity_summary['occupancy_range'].append(sensitivity['occupancy_range'])
            sensitivity_summary['impedance_range'].append(sensitivity['impedance_range'])
            sensitivity_summary['occupancy_topk_flip_rate'].append(sensitivity.get('occupancy_topk_flip_rate', None))
            
            # Print sensitivity for this dimension
            print(f"Sensitivity analysis for dimension {dim_idx}:")
            print(f"  Heatmap  - Variance: {sensitivity['heatmap_variance']:.6f}, Range: {sensitivity['heatmap_range']:.4f}")
            print(f"  Occupancy- Variance: {sensitivity['occupancy_variance']:.6f}, Range: {sensitivity['occupancy_range']:.4f}")
            if sensitivity.get('occupancy_topk_flip_rate', None) is not None:
                print(f"            Top-K flip rate (K={K_VALUE}): {sensitivity['occupancy_topk_flip_rate']:.3f}")
            print(f"  Impedance- Variance: {sensitivity['impedance_variance']:.6f}, Range: {sensitivity['impedance_range']:.4f}")
        
        # Create summary dashboard
        self.create_summary_dashboard(sensitivity_summary, dimensions)
        
        # Save complete results
        self.save_analysis_results(all_results, sensitivity_summary)
        
        return all_results
    
    def create_summary_dashboard(self, sensitivity_summary: Dict, dimensions: List[int]) -> None:
        """Create a summary dashboard showing all dimensions"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot variance comparison
        ax = axes[0, 0]
        x = np.array(sensitivity_summary['dimension'])
        ax.plot(x, sensitivity_summary['heatmap_variance'], 'o-', label='Heatmap', color='orange')
        ax.plot(x, sensitivity_summary['occupancy_variance'], 's-', label='Occupancy', color='purple')
        ax.plot(x, sensitivity_summary['impedance_variance'], '^-', label='Impedance', color='brown')
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Variance')
        ax.set_title('Modality Sensitivity (Variance) Across Dimensions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot range comparison
        ax = axes[0, 1]
        ax.plot(x, sensitivity_summary['heatmap_range'], 'o-', label='Heatmap', color='orange')
        ax.plot(x, sensitivity_summary['occupancy_range'], 's-', label='Occupancy', color='purple')
        ax.plot(x, sensitivity_summary['impedance_range'], '^-', label='Impedance', color='brown')
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Range (Max - Min)')
        ax.set_title('Modality Sensitivity (Range) Across Dimensions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Heatmap of all sensitivities
        ax = axes[1, 0]
        sensitivity_matrix = np.array([
            sensitivity_summary['heatmap_variance'],
            sensitivity_summary['occupancy_variance'], 
            sensitivity_summary['impedance_variance']
        ])
        
        im = ax.imshow(sensitivity_matrix, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(dimensions)))
        ax.set_xticklabels(dimensions)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Heatmap', 'Occupancy', 'Impedance'])
        ax.set_xlabel('Latent Dimension')
        ax.set_title('Sensitivity Heatmap (Variance)')
        plt.colorbar(im, ax=ax)
        
        # Top influencing dimensions per modality
        ax = axes[1, 1]

        top_n = min(3, len(dimensions))
        hm_top = np.argsort(sensitivity_summary['heatmap_variance'])[-top_n:][::-1]
        occ_top = np.argsort(sensitivity_summary['occupancy_variance'])[-top_n:][::-1]
        imp_top = np.argsort(sensitivity_summary['impedance_variance'])[-top_n:][::-1]

        def _fmt_block(title: str, idxs: np.ndarray, key: str) -> str:
            lines = [f"{title}:"]
            for rank, j in enumerate(list(idxs), start=1):
                dim = dimensions[int(j)]
                var = sensitivity_summary[key][int(j)]
                lines.append(f"  {rank}. Dim {dim} (var: {var:.4f})")
            return "\n".join(lines)

        text_summary = "\n".join([
            "Top Influential Dimensions:",
            "",
            _fmt_block("Heatmap", hm_top, "heatmap_variance"),
            "",
            _fmt_block("Occupancy", occ_top, "occupancy_variance"),
            "",
            _fmt_block("Impedance", imp_top, "impedance_variance"),
            "",
        ])
        
        ax.text(0.05, 0.95, text_summary, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Most Influential Dimensions per Modality')
        
        plt.suptitle('Latent Space Analysis Summary', fontsize=16)
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = self.output_dir / 'latent_analysis_summary.png'
        plt.savefig(dashboard_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"\\nSaved summary dashboard: {dashboard_path}")
        
        # Print key findings
        print(f"\\n{'='*60}")
        print("KEY FINDINGS:")
        print(f"{'='*60}")
        print(text_summary)
    
    def save_analysis_results(self, all_results: Dict, sensitivity_summary: Dict) -> None:
        """Save analysis results to files"""
        
        # Save sensitivity summary as JSON
        summary_path = self.output_dir / 'sensitivity_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(sensitivity_summary, f, indent=2)
        
        print(f"Saved sensitivity summary: {summary_path}")
        
        # Save a detailed text report
        report_path = self.output_dir / 'analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write("LATENT TRAVERSAL ANALYSIS REPORT\\n")
            f.write("="*50 + "\\n\\n")
            
            f.write(f"Model: {CHECKPOINT_PATH}\\n")
            f.write(f"Latent dimensions: {LATENT_DIM}\\n")
            f.write(f"Traversal range: {TRAVERSAL_RANGE}\\n")
            f.write(f"Number of steps: {NUM_STEPS}\\n")
            f.write(f"Analyzed dimensions: {DIMENSIONS_TO_ANALYZE}\\n\\n")
            
            f.write("SENSITIVITY ANALYSIS:\\n")
            f.write("-" * 30 + "\\n")
            
            for i, dim in enumerate(sensitivity_summary['dimension']):
                f.write(f"\\nDimension {dim}:\\n")
                f.write(f"  Heatmap  - Variance: {sensitivity_summary['heatmap_variance'][i]:.6f}, ")
                f.write(f"Range: {sensitivity_summary['heatmap_range'][i]:.4f}\\n")
                f.write(f"  Occupancy- Variance: {sensitivity_summary['occupancy_variance'][i]:.6f}, ")
                f.write(f"Range: {sensitivity_summary['occupancy_range'][i]:.4f}\\n")
                f.write(f"  Impedance- Variance: {sensitivity_summary['impedance_variance'][i]:.6f}, ")
                f.write(f"Range: {sensitivity_summary['impedance_range'][i]:.4f}\\n")
        
        print(f"Saved detailed report: {report_path}")

    def _occupancy_vec_to_grid(self, occ_vec: np.ndarray) -> np.ndarray:
        """Convert a (52,) occupancy vector to a (7, 8) grid with invalid cells set to 0."""
        occ_vec = np.asarray(occ_vec).reshape(-1)
        grid = np.zeros((7, 8), dtype=float)

        grid_dict, _ = build_occupancy_grid_dict()
        valid_coords = list(grid_dict.keys())
        if occ_vec.shape[0] != len(valid_coords):
            raise ValueError(
                f"Expected occupancy vector length {len(valid_coords)} (valid cells), got {occ_vec.shape[0]}"
            )

        for i, (r, c) in enumerate(valid_coords):
            grid[r, c] = float(occ_vec[i])
        return grid

    def _topk_binary(self, occ_vec: np.ndarray, k: int) -> np.ndarray:
        """Return a binary (52,) vector with exactly k ones at the highest-prob indices."""
        occ_vec = np.asarray(occ_vec).reshape(-1)
        if k <= 0:
            return np.zeros_like(occ_vec, dtype=float)
        k_eff = min(int(k), occ_vec.shape[0])
        idx = np.argpartition(-occ_vec, k_eff - 1)[:k_eff]
        out = np.zeros_like(occ_vec, dtype=float)
        out[idx] = 1.0
        return out


def main():
    """Main execution function"""
    print("="*60)
    print("LATENT TRAVERSAL ANALYSIS FOR MULTI-INPUT VAE")
    print("="*60)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Latent dimensions: {LATENT_DIM}")
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Traversal range: {TRAVERSAL_RANGE}")
    print(f"Steps per dimension: {NUM_STEPS}")
    print(f"Dimensions to analyze: {DIMENSIONS_TO_ANALYZE}")
    
    # Initialize analyzer
    analyzer = LatentTraversalAnalyzer(
        checkpoint_path=CHECKPOINT_PATH,
        latent_dim=LATENT_DIM,
        device=DEVICE
    )
    
    # Run complete analysis
    results = analyzer.run_full_analysis(DIMENSIONS_TO_ANALYZE)
    
    print(f"\\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"✓ Generated {len(DIMENSIONS_TO_ANALYZE)} individual traversal plots")
    print(f"✓ Created summary dashboard")
    print(f"✓ Saved sensitivity analysis data")
    print(f"✓ Generated detailed report")
    print(f"\\nAll outputs saved to: {OUTPUT_DIR}")
    
    # Final recommendations
    print(f"\\n{'='*60}")
    print("RECOMMENDATIONS:")
    print(f"{'='*60}")
    print("1. Check individual traversal plots to see how each modality responds")
    print("2. Examine the summary dashboard for cross-dimensional comparisons") 
    print("3. Focus on dimensions with high sensitivity for targeted generation")
    print("4. Use low-sensitivity dimensions for fine-tuning without major changes")
    print("5. Consider the top influential dimensions for each modality")


if __name__ == "__main__":
    main()