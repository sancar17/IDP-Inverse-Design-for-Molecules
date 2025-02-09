import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from typing import Tuple, List, Any, Dict
import rdkit.RDLogger as RDLogger
from rdkit import Chem
from fast_jtnn import *
from fast_jtnn.joint_model_v3_3p import JTNNVAE_joint
import matplotlib.pyplot as plt
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import io
from rdkit.Chem import Draw
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def plot_2d_direct_property_predictions(model, samples, prop_indices: Dict, 
                                      prop_means: Dict, prop_stds: Dict,
                                      target_intervals: Dict, property_names: Tuple[str, str], 
                                      save_path: str, dataset_samples: Tuple[np.ndarray, np.ndarray]):
    """Plot 2D visualization of property predictions with dataset distribution and target box"""
    with torch.no_grad():
        out = model.propNN(samples)
        y_pred1 = out[:, prop_indices[property_names[0]]] * prop_stds[property_names[0]] + prop_means[property_names[0]]
        y_pred2 = out[:, prop_indices[property_names[1]]] * prop_stds[property_names[1]] + prop_means[property_names[1]]
        predictions = np.column_stack((y_pred1.cpu().numpy(), y_pred2.cpu().numpy()))

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot dataset distribution as scatter
    ax.scatter(dataset_samples[0], dataset_samples[1], 
              color='red', alpha=0.03, label='QM9 Dataset')
    
    # Plot predictions distribution
    scatter = ax.scatter(predictions[:, 0], predictions[:, 1], 
                        alpha=0.3, c='blue', label='Posterior Samples')
    
    # Plot target box
    target_box = Rectangle(
        (target_intervals[property_names[0]][0], target_intervals[property_names[1]][0]),
        target_intervals[property_names[0]][1] - target_intervals[property_names[0]][0],
        target_intervals[property_names[1]][1] - target_intervals[property_names[1]][0],
        fill=False, color='purple', linewidth=2, label='Target Region'
    )
    ax.add_patch(target_box)
    
    ax.set_xlabel(property_names[0])
    ax.set_ylabel(property_names[1])
    ax.set_title(f'{property_names[0]} vs {property_names[1]} Posterior Distribution')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return predictions

def plot_log_likelihood(trajectories: List[List[float]], save_path: str):
    """Plot all MCMC trajectory log likelihoods"""
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
    
    for i, traj in enumerate(trajectories):
        plt.plot(traj, color=colors[i], alpha=0.3)
    
    plt.xlabel('Trajectory Step')
    plt.ylabel('Combined Log Likelihood')
    plt.title('Log Likelihood Evolution During Sampling')
    plt.ylim(-30, 1)
    plt.savefig(save_path)
    plt.close()

def create_molecule_grid(mols: List, predictions: List[Tuple[float, float]], 
                        dataset_values: List[Tuple[float, float]], 
                        property_names: Tuple[str, str],
                        target_intervals: Dict,
                        save_path: str, n_cols: int = 5):
    """Create a grid of molecule visualizations with 2D predictions"""
    n_mols = len(mols)
    n_rows = (n_mols + n_cols - 1) // n_cols
    
    mol_size = (300, 300)
    img_size = (mol_size[0] * n_cols, mol_size[1] * n_rows + 40)
    
    grid = Image.new('RGB', img_size, 'white')
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    title = (f"{property_names[0]}, {property_names[1]} - Target: "
             f"({target_intervals[property_names[0]][0]:.3f}-{target_intervals[property_names[0]][1]:.3f}, "
             f"{target_intervals[property_names[1]][0]:.3f}-{target_intervals[property_names[1]][1]:.3f})")
    draw.text((10, 10), title, font=font, fill='black')
    
    for idx, (mol, pred_pair, real_pair) in enumerate(zip(mols, predictions, dataset_values)):
        mol_copy = Chem.Mol(mol)
        
        try:
            mol_copy.GetConformer()
        except:
            AllChem.Compute2DCoords(mol_copy)
        
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(mol_size[0], mol_size[1])
        drawer.drawOptions().clearBackground = True
        drawer.drawOptions().backgroundColour = (1, 1, 1)
        drawer.drawOptions().bondLineWidth = 2
        
        drawer.DrawMolecule(mol_copy)
        drawer.FinishDrawing()
        
        png_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png_data))
        
        row = idx // n_cols
        col = idx % n_cols
        x_offset = col * mol_size[0]
        y_offset = row * mol_size[1] + 40
        
        # White background for text
        draw.rectangle([x_offset, y_offset, x_offset + mol_size[0], y_offset + 45], 
                      fill='white')
        draw.rectangle([x_offset, y_offset + mol_size[1] - 45, x_offset + mol_size[0], 
                       y_offset + mol_size[1]], fill='white')
        
        grid.paste(img, (x_offset, y_offset))
        
        # Add predictions for both properties
        pred_text = f"Pred: ({pred_pair[0]:.3f}, {pred_pair[1]:.3f})"
        draw.text((x_offset + 5, y_offset + 5), pred_text, font=font, fill='black')
        
        if real_pair[0] is not None and real_pair[1] is not None:
            val_text = f"Real: ({real_pair[0]:.3f}, {real_pair[1]:.3f})"
        else:
            val_text = "Real: N/A"
        draw.text((x_offset + 5, y_offset + mol_size[1] - 40), 
                 val_text, font=font, fill='black')
        
        draw.rectangle([x_offset, y_offset, x_offset + mol_size[0], 
                       y_offset + mol_size[1]], outline='black', width=1)
    
    grid.save(save_path)

def decode_encode_posterior_samples(model, posterior_samples, prop_indices: Dict,
                                  prop_means: Dict, prop_stds: Dict, 
                                  target_intervals: Dict,
                                  property_names: Tuple[str, str]):
    """Decode and re-encode posterior samples with 2D property predictions"""
    valid_mols = []
    predictions = []
    log_likelihoods = []
    smiles_set = set()
    
    with torch.no_grad():
        out = model.propNN(posterior_samples)
        y_pred1 = out[:, prop_indices[property_names[0]]] * prop_stds[property_names[0]] + prop_means[property_names[0]]
        y_pred2 = out[:, prop_indices[property_names[1]]] * prop_stds[property_names[1]] + prop_means[property_names[1]]
        y_pred_sigma1 = torch.exp(out[:, prop_indices[property_names[0]] + 3]) * prop_stds[property_names[0]]
        y_pred_sigma2 = torch.exp(out[:, prop_indices[property_names[1]] + 3]) * prop_stds[property_names[1]]
        
        # Calculate combined log likelihood using CDFs for intervals
        normal_dist1 = dist.Normal(y_pred1, y_pred_sigma1)
        normal_dist2 = dist.Normal(y_pred2, y_pred_sigma2)
        
        # Calculate interval probabilities using CDFs
        cdf_upper1 = normal_dist1.cdf(torch.tensor(target_intervals[property_names[0]][1]).cuda())
        cdf_lower1 = normal_dist1.cdf(torch.tensor(target_intervals[property_names[0]][0]).cuda())
        cdf_upper2 = normal_dist2.cdf(torch.tensor(target_intervals[property_names[1]][1]).cuda())
        cdf_lower2 = normal_dist2.cdf(torch.tensor(target_intervals[property_names[1]][0]).cuda())
        
        # Combined log likelihood for intervals
        batch_log_likes = torch.log(cdf_upper1 - cdf_lower1) + torch.log(cdf_upper2 - cdf_lower2)

        decode_times = 5
        
        for i, z in enumerate(posterior_samples):
            for j in range(decode_times):
                z_tree = torch.zeros(1, 28).cuda()
                z_mol = torch.zeros(1, 28).cuda()
                z_mol[:, :7] = z[:7]
                z_tree[:, :7] = z[7:]
                
                smiles = model.decode(z_tree, z_mol, prob_decode=False)
                if smiles and smiles not in smiles_set:
                    smiles_set.add(smiles)
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            valid_mols.append(mol)
                            predictions.append((y_pred1[i].item(), y_pred2[i].item()))
                            log_likelihoods.append(batch_log_likes[i].item())
                    except:
                        continue
    
    return valid_mols, predictions, log_likelihoods

def select_best_molecules_2d(valid_mols: List, predictions: List[Tuple[float, float]], 
                           log_likelihoods: List[float], target_intervals: Dict,
                           property_names: Tuple[str, str], k: int = 20) -> Tuple[List, List[Tuple[float, float]]]:
    """Select top k molecules based on whether they're in the target box and their log likelihood"""
    pred_array = np.array(predictions)
    
    # Check which predictions fall within the target intervals
    in_box = np.logical_and.reduce([
        pred_array[:, 0] >= target_intervals[property_names[0]][0],
        pred_array[:, 0] <= target_intervals[property_names[0]][1],
        pred_array[:, 1] >= target_intervals[property_names[1]][0],
        pred_array[:, 1] <= target_intervals[property_names[1]][1]
    ])
    
    # Combine with log likelihood scores
    likelihood_scores = np.array(log_likelihoods)
    
    # Prioritize molecules in the box, then by likelihood
    combined_scores = in_box.astype(float) * 1000 + likelihood_scores
    
    # Select top k based on combined score
    top_indices = np.argsort(combined_scores)[-k:]
    
    return [valid_mols[i] for i in top_indices], [predictions[i] for i in top_indices]

def sample_posterior_pair(model, target_intervals: Dict, prop_indices: Dict,
                         prop_means: Dict, prop_stds: Dict,
                         property_names: Tuple[str, str]):
    """Sample from posterior considering both properties simultaneously using intervals"""
    # Normalize target intervals
    intervals_normalized = {
        prop: (
            torch.tensor([(val[0] - prop_means[prop]) / prop_stds[prop]], 
                        dtype=torch.float32).cuda(),
            torch.tensor([(val[1] - prop_means[prop]) / prop_stds[prop]], 
                        dtype=torch.float32).cuda()
        )
        for prop, val in target_intervals.items()
    }
    
    all_trajectories = []
    current_trajectory = []
    
    def model_fn():
        # Sample latent vector
        z_prop = pyro.sample(
            "z_prop",
            dist.Normal(torch.zeros(14, dtype=torch.float32).cuda(), 
                       torch.ones(14, dtype=torch.float32).cuda())
        ).unsqueeze(0)
        
        # Get predictions and uncertainties
        out = model.propNN(z_prop)
        y_mu1 = out[:, prop_indices[property_names[0]]]
        y_mu2 = out[:, prop_indices[property_names[1]]]
        y_sigma1 = torch.exp(out[:, prop_indices[property_names[0]] + 3])
        y_sigma2 = torch.exp(out[:, prop_indices[property_names[1]] + 3])
        
        # Sample uniformly from target intervals
        y1 = pyro.sample("y1", dist.Uniform(
            intervals_normalized[property_names[0]][0],
            intervals_normalized[property_names[0]][1]
        ))
        y2 = pyro.sample("y2", dist.Uniform(
            intervals_normalized[property_names[1]][0],
            intervals_normalized[property_names[1]][1]
        ))
        
        # Calculate probabilities of being in intervals under the model
        normal1 = dist.Normal(y_mu1, y_sigma1)
        normal2 = dist.Normal(y_mu2, y_sigma2)
        
        prob1 = normal1.cdf(intervals_normalized[property_names[0]][1]) - \
                normal1.cdf(intervals_normalized[property_names[0]][0])
        prob2 = normal2.cdf(intervals_normalized[property_names[1]][1]) - \
                normal2.cdf(intervals_normalized[property_names[1]][0])
        
        # Combined log likelihood
        log_like = torch.log(prob1) + torch.log(prob2)
        current_trajectory.append(log_like.item())
        
        return torch.cat([y1, y2])
    
    nuts_kernel = NUTS(model_fn, 
                      adapt_step_size=True,
                      target_accept_prob=0.9,
                      max_tree_depth=15,
                      step_size=0.01,  # Smaller initial step size
                      adapt_mass_matrix=True)
    
    mcmc = MCMC(nuts_kernel, num_samples=3000, warmup_steps=1000)
    
    original_sample = nuts_kernel.sample
    def sample_wrapper(*args, **kwargs):
        nonlocal current_trajectory
        result = original_sample(*args, **kwargs)
        if current_trajectory:
            all_trajectories.append(current_trajectory)
            current_trajectory = []
        return result
    nuts_kernel.sample = sample_wrapper
    
    mcmc.run()
    samples = mcmc.get_samples()["z_prop"]
    
    return samples, mcmc, all_trajectories

def plot_2d_selected_property_predictions(selected_preds: List[Tuple[float, float]], 
                                        target_intervals: Dict, 
                                        property_names: Tuple[str, str],
                                        save_path: str,
                                        dataset_samples: Tuple[np.ndarray, np.ndarray]):
    """Plot 2D distribution of selected molecules' properties"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot dataset as scatter
    ax.scatter(dataset_samples[0], dataset_samples[1], 
              color='red', alpha=0.03, label='QM9 Dataset')
    
    # Convert predictions to array for plotting
    selected_array = np.array(selected_preds)
    
    # Plot selected molecules
    ax.scatter(selected_array[:, 0], selected_array[:, 1], 
              color='green', alpha=0.3, label='Selected Molecules')
    
    # Plot target box
    target_box = Rectangle(
        (target_intervals[property_names[0]][0], target_intervals[property_names[1]][0]),
        target_intervals[property_names[0]][1] - target_intervals[property_names[0]][0],
        target_intervals[property_names[1]][1] - target_intervals[property_names[1]][0],
        fill=False, color='purple', linewidth=2, label='Target Region'
    )
    ax.add_patch(target_box)
    
    ax.set_xlabel(property_names[0])
    ax.set_ylabel(property_names[1])
    ax.set_title(f'{property_names[0]} vs {property_names[1]} Selected Molecules Distribution')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_molecule_data(mols: List, predictions: List[Tuple[float, float]], 
                      property_names: Tuple[str, str], 
                      target_intervals: Dict, entire_data: pd.DataFrame,
                      save_path: str) -> pd.DataFrame:
    """Save molecule data with 2D property predictions"""
    data = []
    for mol, pred in zip(mols, predictions):
        smiles = Chem.MolToSmiles(mol)
        exists = smiles in set(entire_data['smiles'])
        
        dataset_values = (
            entire_data[entire_data['smiles'] == smiles][property_names[0]].iloc[0] if exists else None,
            entire_data[entire_data['smiles'] == smiles][property_names[1]].iloc[0] if exists else None
        )
        
        # Check if prediction is within target intervals
        in_target_region = (
            target_intervals[property_names[0]][0] <= pred[0] <= target_intervals[property_names[0]][1] and
            target_intervals[property_names[1]][0] <= pred[1] <= target_intervals[property_names[1]][1]
        )
        
        data.append({
            'smiles': smiles,
            f'predicted_{property_names[0]}': pred[0],
            f'predicted_{property_names[1]}': pred[1],
            'exists_in_dataset': exists,
            f'dataset_{property_names[0]}': dataset_values[0],
            f'dataset_{property_names[1]}': dataset_values[1],
            'in_target_region': in_target_region,
            'distance_to_target_center': np.sqrt(
                (pred[0] - np.mean(target_intervals[property_names[0]]))**2 +
                (pred[1] - np.mean(target_intervals[property_names[1]]))**2
            )
        })
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    return df

def process_property_pair(model: JTNNVAE_joint,
                         property_names: Tuple[str, str],
                         target_intervals: Dict,
                         prop_indices: Dict,
                         prop_means: Dict,
                         prop_stds: Dict,
                         analysis_dir: str,
                         entire_data: pd.DataFrame):
    """Process a pair of properties together with 2D analysis"""
    plots_dir = os.path.join(analysis_dir, 'plots')
    molecules_dir = os.path.join(analysis_dir, 'molecule_visualizations')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(molecules_dir, exist_ok=True)
    
    # Get dataset samples for both properties
    dataset_samples = (
        entire_data[property_names[0]].values,
        entire_data[property_names[1]].values
    )
    
    # Create filename suffix with target intervals
    target_suffix = (f"_{target_intervals[property_names[0]][0]:.3f}"
                    f"_{target_intervals[property_names[0]][1]:.3f}"
                    f"_{target_intervals[property_names[1]][0]:.3f}"
                    f"_{target_intervals[property_names[1]][1]:.3f}")
    
    print(f"Processing property pair: {property_names[0]}-{property_names[1]}")
    print(f"Target intervals: {target_intervals}")
    
    # Sample from posterior
    samples, mcmc, log_likelihoods = sample_posterior_pair(
        model, target_intervals, prop_indices, prop_means, prop_stds, property_names
    )
    
    # Plot log likelihood evolution
    plot_log_likelihood(
        log_likelihoods,
        f'{plots_dir}/{property_names[0]}_{property_names[1]}_loglikelihood{target_suffix}.png'
    )
    
    # Get direct property predictions
    predictions = plot_2d_direct_property_predictions(
        model, samples,
        prop_indices, prop_means, prop_stds,
        target_intervals, property_names,
        f'{plots_dir}/{property_names[0]}_{property_names[1]}_direct_predictions{target_suffix}.png',
        dataset_samples
    )
    
    # Decode and analyze molecules
    valid_mols, mol_predictions, mol_log_likelihoods = decode_encode_posterior_samples(
        model, samples, prop_indices, prop_means, prop_stds,
        target_intervals, property_names
    )
    
    # Select best molecules
    best_mols, best_preds = select_best_molecules_2d(
        valid_mols, mol_predictions, mol_log_likelihoods,
        target_intervals, property_names
    )
    
    # Plot selected molecules' properties
    plot_2d_selected_property_predictions(
        best_preds, target_intervals, property_names,
        f'{plots_dir}/{property_names[0]}_{property_names[1]}_selected_predictions{target_suffix}.png',
        dataset_samples
    )
    
    # Get dataset values for visualization
    dataset_values = []
    for mol in best_mols:
        smiles = Chem.MolToSmiles(mol)
        if smiles in set(entire_data['smiles']):
            val1 = entire_data[entire_data['smiles'] == smiles][property_names[0]].iloc[0]
            val2 = entire_data[entire_data['smiles'] == smiles][property_names[1]].iloc[0]
            dataset_values.append((val1, val2))
        else:
            dataset_values.append((None, None))
    
    # Create molecule visualization grid
    create_molecule_grid(
        best_mols, best_preds, dataset_values,
        property_names,
        target_intervals,
        f'{molecules_dir}/{property_names[0]}_{property_names[1]}_molecules{target_suffix}.png'
    )
    
    # Save molecule data
    df = save_molecule_data(
        best_mols, best_preds,
        property_names, target_intervals,
        entire_data,
        f'{analysis_dir}/{property_names[0]}_{property_names[1]}_molecules{target_suffix}.csv'
    )
    
    return {
        'predictions': predictions,
        'selected_mols': best_mols,
        'selected_preds': best_preds,
        'mcmc': mcmc
    }

def main():
    # Paths and initialization
    vocab_path = '../../data/vocab.txt'
    model_path = '../../joint_training/entire_data_after_error_check/model.best'
    analysis_dir = './box_sampling_results_modelv3/'
    entire_data = pd.read_csv('../../data/qm9_smiles_prop_normalized.csv')

    # Calculate means and standard deviations
    property_means = {
        'homo': entire_data['homo'].mean(),
        'lumo': entire_data['lumo'].mean(),
        'r2': entire_data['r2'].mean()
    }
    
    property_stds = {
        'homo': entire_data['homo'].std(),
        'lumo': entire_data['lumo'].std(),
        'r2': entire_data['r2'].std()
    }
    
    property_indices = {
        'homo': 0,
        'lumo': 1,
        'r2': 2
    }

    # Define property pairs and their target intervals
    property_pairs = [
        (('homo', 'lumo'), {'homo': (-0.27, -0.23), 'lumo': (-0.02, 0.02)}),
        (('homo', 'lumo'), {'homo': (-0.22, -0.18), 'lumo': (-0.07, -0.03)}),
        (('homo', 'r2'), {'homo': (-0.26, -0.22), 'r2': (700, 900)}),
        (('homo', 'r2'), {'homo': (-0.22, -0.18), 'r2': (1100, 1300)}),
        (('lumo', 'r2'), {'lumo': (-0.02, 0.02), 'r2': (700, 900)}),
        (('lumo', 'r2'), {'lumo': (0.06, 0.10), 'r2': (1400, 1600)})
    ]

    # Create output directories
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Load vocabulary and model
    vocab = [x.strip("\r\n ") for x in open(vocab_path)]
    vocab = Vocab(vocab)
    
    hidden_size = 450
    latent_size = 56
    depthT = 20
    depthG = 3
    
    model = JTNNVAE_joint(vocab, hidden_size, latent_size, depthT, depthG, property_weight=1, z_prop_size=14)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    
    # Process each property pair
    results = {}
    for prop_names, target_intervals in property_pairs:
        results[f"{prop_names[0]}_{prop_names[1]}"] = process_property_pair(
            model=model,
            property_names=prop_names,
            target_intervals=target_intervals,
            prop_indices=property_indices,
            prop_means=property_means,
            prop_stds=property_stds,
            analysis_dir=analysis_dir,
            entire_data=entire_data
        )
    
    return results

if __name__ == "__main__":
    results = main()