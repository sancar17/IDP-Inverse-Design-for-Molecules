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
from typing import Tuple, List, Any, Dict
import rdkit.RDLogger as RDLogger
from rdkit import Chem
from fast_jtnn import *
from fast_jtnn.joint_model_v1_3p import JTNNVAE_joint
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
                                      target_values: Dict, property_names: Tuple[str, str], 
                                      save_path: str, dataset_samples: Tuple[np.ndarray, np.ndarray]):
    """Plot 2D visualization of property predictions with dataset distribution"""
    with torch.no_grad():
        out = model.propNN(samples)
        y_pred1 = out[:, prop_indices[property_names[0]]] * prop_stds[property_names[0]] + prop_means[property_names[0]]
        y_pred2 = out[:, prop_indices[property_names[1]]] * prop_stds[property_names[1]] + prop_means[property_names[1]]
        predictions = np.column_stack((y_pred1.cpu().numpy(), y_pred2.cpu().numpy()))

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot dataset distribution as scatter
    ax.scatter(dataset_samples[0], dataset_samples[1], 
              color='red', alpha=0.03, label='QM9 Dataset')
    
    # Plot predictions distribution with viridis colormap
    scatter = ax.scatter(predictions[:, 0], predictions[:, 1], 
                        alpha=0.3, c='blue', label='Posterior Samples')
    
    # Plot target point with larger marker and black edge
    ax.scatter([target_values[property_names[0]]], [target_values[property_names[1]]], 
               color='purple', s=100, marker='*', edgecolor='black',
               label='Target', zorder=5)
    
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
                        target_values: Tuple[float, float],
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
    
    title = f"{property_names[0]}, {property_names[1]} - Target: ({target_values[0]:.3f}, {target_values[1]:.3f})"
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
                                  target_values: Dict,
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
        
        # Calculate combined log likelihood
        normal_dist1 = dist.Normal(y_pred1, y_pred_sigma1)
        normal_dist2 = dist.Normal(y_pred2, y_pred_sigma2)
        target_tensor1 = torch.full_like(y_pred1, target_values[property_names[0]])
        target_tensor2 = torch.full_like(y_pred2, target_values[property_names[1]])
        batch_log_likes = normal_dist1.log_prob(target_tensor1) + normal_dist2.log_prob(target_tensor2)

        decode_times = 1
        
        for i, z in enumerate(posterior_samples):
            for j in range(decode_times):
                z_tree = torch.zeros(1, 28).cuda()
                z_mol = torch.zeros(1, 28).cuda()
                z_mol[:, :] = z[:28]
                z_tree[:, :] = z[28:]
                
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
                           log_likelihoods: List[float], target_values: Tuple[float, float], 
                           k: int = 20) -> Tuple[List, List[Tuple[float, float]]]:
    """Select top k molecules based on 2D property distance and log likelihood"""
    # Calculate distances to target in normalized 2D property space
    pred_array = np.array(predictions)
    target_array = np.array(target_values)
    
    # Normalize distances by property ranges
    prop_ranges = np.ptp(pred_array, axis=0)
    normalized_preds = pred_array / prop_ranges
    normalized_target = target_array / prop_ranges
    
    distances = cdist(normalized_preds.reshape(-1, 2), 
                     normalized_target.reshape(1, 2)).flatten()
    
    # Combine distance and likelihood scores
    distance_scores = -distances  # Negative because we want to maximize
    likelihood_scores = np.array(log_likelihoods)
    
    # Normalize scores
    distance_scores = (distance_scores - np.mean(distance_scores)) / np.std(distance_scores)
    likelihood_scores = (likelihood_scores - np.mean(likelihood_scores)) / np.std(likelihood_scores)
    
    # Combined score with equal weighting
    combined_scores = distance_scores + likelihood_scores
    
    # Select top k based on combined score
    top_indices = np.argsort(combined_scores)[-k:]
    
    return [valid_mols[i] for i in top_indices], [predictions[i] for i in top_indices]

def sample_posterior_pair(model, target_values: Dict, prop_indices: Dict,
                         prop_means: Dict, prop_stds: Dict,
                         property_names: Tuple[str, str]):
    """Sample from posterior considering both properties simultaneously"""
    y_target1 = (torch.tensor([target_values[property_names[0]]]).cuda() - prop_means[property_names[0]]) / prop_stds[property_names[0]]
    y_target2 = (torch.tensor([target_values[property_names[1]]]).cuda() - prop_means[property_names[1]]) / prop_stds[property_names[1]]
    
    all_trajectories = []
    current_trajectory = []
    
    def model_fn(y_obs1, y_obs2):
        z_prop = pyro.sample(
            "z_prop",
            dist.Normal(torch.zeros(56).cuda(), torch.ones(56).cuda())
        ).unsqueeze(0)
        
        out = model.propNN(z_prop)
        y_mu1 = out[:, prop_indices[property_names[0]]]
        y_mu2 = out[:, prop_indices[property_names[1]]]
        y_sigma1 = torch.exp(out[:, prop_indices[property_names[0]] + 3])
        y_sigma2 = torch.exp(out[:, prop_indices[property_names[1]] + 3])
        
        with pyro.plate("data"):
            likelihood1 = dist.Normal(y_mu1, y_sigma1)
            likelihood2 = dist.Normal(y_mu2, y_sigma2)
            
            obs1 = pyro.sample("obs1", likelihood1, obs=y_obs1)
            obs2 = pyro.sample("obs2", likelihood2, obs=y_obs2)
            
            current_trajectory.append(likelihood1.log_prob(y_obs1).item() + likelihood2.log_prob(y_obs2).item())
    
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
    
    mcmc.run(y_obs1=y_target1, y_obs2=y_target2)
    samples = mcmc.get_samples()["z_prop"]
    
    return samples, mcmc, all_trajectories


def plot_2d_selected_property_predictions(selected_preds: List[Tuple[float, float]], 
                                        target_values: Dict, 
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
    
    # Plot selected molecules with viridis colormap
    ax.scatter(selected_array[:, 0], selected_array[:, 1], 
              color='green', alpha=0.3, label='Selected Molecules')
    
    # Plot target point with larger marker and black edge
    ax.scatter([target_values[property_names[0]]], [target_values[property_names[1]]], 
               color='purple', s=100, marker='*', edgecolor='black',
               label='Target', zorder=5)
    
    ax.set_xlabel(property_names[0])
    ax.set_ylabel(property_names[1])
    ax.set_title(f'{property_names[0]} vs {property_names[1]} Selected Molecules Distribution')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_molecule_data(mols: List, predictions: List[Tuple[float, float]], 
                      property_names: Tuple[str, str], 
                      target_values: Dict, entire_data: pd.DataFrame,
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
        
        data.append({
            'smiles': smiles,
            f'predicted_{property_names[0]}': pred[0],
            f'predicted_{property_names[1]}': pred[1],
            'exists_in_dataset': exists,
            f'dataset_{property_names[0]}': dataset_values[0],
            f'dataset_{property_names[1]}': dataset_values[1],
            'distance_to_target': np.sqrt(
                (pred[0] - target_values[property_names[0]])**2 +
                (pred[1] - target_values[property_names[1]])**2
            )
        })
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    return df

def process_property_pair(model: JTNNVAE_joint,
                         property_names: Tuple[str, str],
                         target_values: Dict,
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
    
    # Create filename suffix with target values
    target_suffix = f"_{target_values[property_names[0]]:.3f}_{target_values[property_names[1]]:.3f}"
    
    print(f"Processing property pair: {property_names[0]}-{property_names[1]}")
    print(f"Target values: {target_values}")
    
    # Sample from posterior
    samples, mcmc, log_likelihoods = sample_posterior_pair(
        model, target_values, prop_indices, prop_means, prop_stds, property_names
    )
    
    # Plot log likelihood evolution with target values in filename
    plot_log_likelihood(
        log_likelihoods,
        f'{plots_dir}/{property_names[0]}_{property_names[1]}_loglikelihood{target_suffix}.png'
    )
    
    # Get direct property predictions with target values in filename
    predictions = plot_2d_direct_property_predictions(
        model, samples,
        prop_indices, prop_means, prop_stds,
        target_values, property_names,
        f'{plots_dir}/{property_names[0]}_{property_names[1]}_direct_predictions{target_suffix}.png',
        dataset_samples
    )
    
    # Decode and analyze molecules
    valid_mols, mol_predictions, mol_log_likelihoods = decode_encode_posterior_samples(
        model, samples, prop_indices, prop_means, prop_stds,
        target_values, property_names
    )
    
    # Select best molecules based on 2D criteria
    best_mols, best_preds = select_best_molecules_2d(
        valid_mols, mol_predictions, mol_log_likelihoods,
        (target_values[property_names[0]], target_values[property_names[1]])
    )
    
    # Plot selected molecules' properties with target values in filename
    plot_2d_selected_property_predictions(
        best_preds, target_values, property_names,
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
    
    # Create molecule visualization grid with target values in filename
    create_molecule_grid(
        best_mols, best_preds, dataset_values,
        property_names,
        (target_values[property_names[0]], target_values[property_names[1]]),
        f'{molecules_dir}/{property_names[0]}_{property_names[1]}_molecules{target_suffix}.png'
    )
    
    # Save molecule data with target values in filename
    df = save_molecule_data(
        best_mols, best_preds,
        property_names, target_values,
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
    model_path = '../../joint_training/v1_3p_entire_data/model.best'
    analysis_dir = './intersection_sampling_results_modelv1/'
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

    # Define property pairs and their target values
    property_pairs = [
        (('homo', 'lumo'), {'homo': -0.23, 'lumo': 0.0}),
        (('homo', 'lumo'), {'homo': -0.27, 'lumo': 0.07}),
        (('homo', 'r2'), {'homo': -0.23, 'r2': 1500}),
        (('homo', 'r2'), {'homo': -0.2, 'r2': 1000}),
        (('lumo', 'r2'), {'lumo': 0.0, 'r2': 1500}),
        (('lumo', 'r2'), {'lumo': -0.07, 'r2': 2000})
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
    
    model = JTNNVAE_joint(vocab, hidden_size, latent_size, depthT, depthG, property_weight=1)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    
    # Process each property pair
    results = {}
    for prop_names, target_values in property_pairs:
        results[f"{prop_names[0]}_{prop_names[1]}"] = process_property_pair(
            model=model,
            property_names=prop_names,
            target_values=target_values,
            prop_indices=property_indices,
            prop_means=property_means,
            prop_stds=property_stds,
            analysis_dir=analysis_dir,
            entire_data=entire_data
        )
    
    return results

if __name__ == "__main__":
    results = main()