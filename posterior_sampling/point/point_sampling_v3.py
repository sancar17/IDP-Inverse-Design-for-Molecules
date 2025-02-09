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
from typing import Tuple, List, Any
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


# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def plot_direct_property_predictions(model, samples, property_index, prop_mean, prop_std, target_value, property_name, save_path, dataset_samples):
    """Plot KDE of property predictions with dataset distribution"""
    with torch.no_grad():
        out = model.propNN(samples)
        y_pred = out[:, property_index] * prop_std + prop_mean
        y_pred = y_pred.cpu().numpy()

    plt.figure(figsize=(10, 6))
    
    # Dataset distribution
    kde_dataset = gaussian_kde(dataset_samples)
    x_dataset = np.linspace(min(dataset_samples), max(dataset_samples), 2048)
    plt.plot(x_dataset, kde_dataset(x_dataset), 'r-', linewidth=2, label='Dataset')
    
    # Predictions distribution
    kde_pred = gaussian_kde(y_pred)
    x_pred = np.linspace(min(y_pred), max(y_pred), 2048)
    plt.plot(x_pred, kde_pred(x_pred), 'b-', linewidth=2, label='Predicted Distribution')
    
    plt.axvline(x=target_value, color='k', linestyle='--', label=f'Target ({target_value:.3f})')
    plt.xlabel(property_name)
    plt.ylabel('Density')
    plt.title(f'{property_name} Predictions Distribution')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    return y_pred

def plot_log_likelihood(trajectories, save_path):
    """Plot all MCMC trajectory log likelihoods"""
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
    
    for i, traj in enumerate(trajectories):
        plt.plot(traj, color=colors[i], alpha=0.3)
    
    plt.xlabel('Trajectory Step')
    plt.ylabel('Log Likelihood')
    plt.title('Log Likelihood Evolution During Sampling')
    plt.ylim(-30, 1)
    plt.savefig(save_path)
    plt.close()

def create_molecule_grid(mols, predictions, dataset_values, property_name, target_value, save_path, n_cols=5):
    """Create a grid of molecule visualizations with predictions and dataset values"""
    n_mols = len(mols)
    n_rows = (n_mols + n_cols - 1) // n_cols
    
    # Calculate image size based on number of molecules
    mol_size = (300, 300)  # Size for each molecule
    img_size = (mol_size[0] * n_cols, mol_size[1] * n_rows + 40)  # Added extra height for title
    
    # Create empty white image
    grid = Image.new('RGB', img_size, 'white')
    draw = ImageDraw.Draw(grid)
    
    # Try to load font, use default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Add title with property and target value at the top
    title = f"{property_name} - Target: {target_value:.3f}"
    draw.text((10, 10), title, font=font, fill='black')
    
    for idx, (mol, pred, real_val) in enumerate(zip(mols, predictions, dataset_values)):
        # Create a copy of the molecule to avoid modifying the original
        mol_copy = Chem.Mol(mol)
        
        # Generate 2D coordinates if they don't exist
        try:
            mol_copy.GetConformer()
        except:
            AllChem.Compute2DCoords(mol_copy)
        
        # Create white background drawing
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(mol_size[0], mol_size[1])
        drawer.drawOptions().clearBackground = True  # Ensure white background
        drawer.drawOptions().backgroundColour = (1, 1, 1)  # White background
        drawer.drawOptions().bondLineWidth = 2
        
        # Draw the molecule
        drawer.DrawMolecule(mol_copy)
        drawer.FinishDrawing()
        
        # Convert to PIL Image
        png_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png_data))
        
        # Calculate position in grid
        row = idx // n_cols
        col = idx % n_cols
        x_offset = col * mol_size[0]
        y_offset = row * mol_size[1] + 40  # Offset by title height
        
        # Create a white rectangle for text background
        draw.rectangle([x_offset, y_offset, x_offset + mol_size[0], y_offset + 25], 
                      fill='white')
        draw.rectangle([x_offset, y_offset + mol_size[1] - 25, x_offset + mol_size[0], 
                      y_offset + mol_size[1]], fill='white')
        
        # Paste molecule image
        grid.paste(img, (x_offset, y_offset))
        
        # Add prediction text on top with white background
        pred_text = f"Pred: {pred:.3f}"
        draw.text((x_offset + 10, y_offset + 5), pred_text, 
                 font=font, fill='black')
        
        # Add dataset value at bottom with white background
        if real_val is not None:
            val_text = f"Real: {real_val:.3f}"
        else:
            val_text = "Real: ?"
        draw.text((x_offset + 10, y_offset + mol_size[1] - 20), 
                 val_text, font=font, fill='black')
        
        # Add a border around the molecule
        draw.rectangle([x_offset, y_offset, x_offset + mol_size[0], 
                       y_offset + mol_size[1]], outline='black', width=1)
    
    # Save the grid
    grid.save(save_path)

def decode_encode_posterior_samples(model, posterior_samples, property_index, prop_mean, prop_std, target_value):
    """Decode and re-encode posterior samples"""
    valid_mols = []
    predictions = []
    log_likelihoods = []
    smiles_set = set()
    
    with torch.no_grad():
        # Initial predictions
        out = model.propNN(posterior_samples)
        y_pred = out[:, property_index] * prop_std + prop_mean
        y_pred_sigma = torch.exp(out[:, property_index + 3]) * prop_std
        
        # Calculate log likelihoods
        normal_dist = dist.Normal(y_pred, y_pred_sigma)
        target_tensor = torch.full_like(y_pred, target_value)
        batch_log_likes = normal_dist.log_prob(target_tensor)

        decode_times = 5
        
        for i, z in enumerate(posterior_samples):
            for j in range(decode_times):
                z_tree = torch.randn(1, 28).cuda()
                z_mol = torch.randn(1, 28).cuda()
                z_mol[:, :7] = z[:7]
                z_tree[:, :7] = z[7:]
                
                smiles = model.decode(z_tree, z_mol, prob_decode=False)
                if smiles and smiles not in smiles_set:
                    smiles_set.add(smiles)
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            valid_mols.append(mol)
                            predictions.append(y_pred[i].item())
                            log_likelihoods.append(batch_log_likes[i].item())
                    except:
                        continue
    
    return valid_mols, predictions, log_likelihoods

def select_best_molecules(valid_mols, predictions, log_likelihoods, k=20):
    """Select top k molecules based on log likelihoods"""
    indices = np.argsort(log_likelihoods)[-k:]
    return [valid_mols[i] for i in indices], [predictions[i] for i in indices]

def save_molecule_data(mols, predictions, property_name, target_value, entire_data, save_path):
    """Save molecule data to CSV"""
    data = []
    for mol, pred in zip(mols, predictions):
        smiles = Chem.MolToSmiles(mol)
        exists = smiles in set(entire_data['smiles'])
        dataset_value = entire_data[entire_data['smiles'] == smiles][property_name].iloc[0] if exists else None
        
        data.append({
            'smiles': smiles,
            'predicted_value': pred,
            'exists_in_dataset': exists,
            'dataset_value': dataset_value
        })
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    return df

def plot_property_kde(dataset_samples, target_samples, target_values, title, label, property_name, save_path):
    """Plot combined distribution showing dataset and all targets"""
    plt.figure(figsize=(10, 6))
    
    # Dataset distribution
    kde = gaussian_kde(dataset_samples)
    x_d = np.linspace(min(dataset_samples), max(dataset_samples), 2048)
    plt.plot(x_d, kde(x_d), color='red', linewidth=2, label="Dataset")
    
    # Target distributions
    cmap = plt.cm.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(target_samples)))
    
    for i, (target_sample_array, target_value) in enumerate(zip(target_samples, target_values)):
        kde = gaussian_kde(target_sample_array)
        density = kde(x_d)
        color = colors[i]
        plt.plot(x_d, density, "--", color=color, linewidth=2, label=f'Target {target_value:.3f}')
        plt.scatter([target_value], [0], color=color, s=300, zorder=5)

    plt.xlabel(property_name)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_selected_property_predictions(best_preds, target_value, property_name, save_path, dataset_samples):
    """Plot KDE of selected molecules' property predictions with dataset distribution"""
    plt.figure(figsize=(10, 6))
    
    # Dataset distribution
    kde_dataset = gaussian_kde(dataset_samples)
    x_dataset = np.linspace(min(dataset_samples), max(dataset_samples), 2048)
    plt.plot(x_dataset, kde_dataset(x_dataset), 'r-', linewidth=2, label='Dataset')
    
    # Selected predictions distribution
    kde_selected = gaussian_kde(best_preds)
    x_selected = np.linspace(min(best_preds), max(best_preds), 2048)
    plt.plot(x_selected, kde_selected(x_selected), 'g-', linewidth=2, label='Selected Molecules')
    
    plt.axvline(x=target_value, color='k', linestyle='--', label=f'Target ({target_value:.3f})')
    plt.xlabel(property_name)
    plt.ylabel('Density')
    plt.title(f'{property_name} Selected Molecules Distribution')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def sample_posterior(model, target_property_value, target_property_index, dataset_mean, dataset_std):
    """Sample from posterior with enhanced diagnostics and log likelihood tracking"""
    y_target = (torch.tensor([target_property_value]).cuda() - dataset_mean) / dataset_std
    all_trajectories = []
    current_trajectory = []
    
    def model_fn(y_obs):
        z_prop = pyro.sample(
            "z_prop",
            dist.Normal(torch.zeros(14).cuda(), torch.ones(14).cuda())
        ).unsqueeze(0)
        
        out = model.propNN(z_prop)
        y_mu, y_sigma_log = out[:, target_property_index], out[:, target_property_index + 3]
        y_sigma = torch.exp(y_sigma_log)
        
        with pyro.plate("data"):
            likelihood = dist.Normal(y_mu, y_sigma)
            obs = pyro.sample("obs", likelihood, obs=y_obs)
            # Store log likelihood
            current_trajectory.append(likelihood.log_prob(y_obs).item())
    
    nuts_kernel = NUTS(model_fn, 
                      adapt_step_size=True,
                      target_accept_prob=0.9,
                      max_tree_depth=15,
                      step_size=0.01,  # Smaller initial step size
                      adapt_mass_matrix=True)
    
    mcmc = MCMC(nuts_kernel, num_samples=3000, warmup_steps=1000)
    mcmc.run(y_obs=y_target)
    
    original_sample = nuts_kernel.sample
    def sample_wrapper(*args, **kwargs):
        nonlocal current_trajectory
        result = original_sample(*args, **kwargs)
        if current_trajectory:
            all_trajectories.append(current_trajectory)
            current_trajectory = []
        return result
    nuts_kernel.sample = sample_wrapper
    
    mcmc.run(y_obs=y_target)
    samples = mcmc.get_samples()["z_prop"]
    
    return samples, mcmc, all_trajectories

def process_property(model, property_name, target_values, property_index, prop_mean, prop_std, analysis_dir, entire_data):
    plots_dir = os.path.join(analysis_dir, 'plots')
    molecules_dir = os.path.join(analysis_dir, 'molecule_visualizations')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(molecules_dir, exist_ok=True)
    
    direct_predictions_list = []
    selected_predictions_list = []
    all_mols = []
    all_diagnostics = []
    
    dataset_samples = entire_data[property_name].values
    
    for target_value in target_values:
        print("Processing:", property_name, " for target value:", target_value)
        
        # Sample from posterior with log likelihood tracking
        samples, mcmc, log_likelihoods = sample_posterior(
            model, target_value, property_index, prop_mean, prop_std
        )
        
        # Plot log likelihood evolution
        plot_log_likelihood(
            log_likelihoods,
            f'{plots_dir}/{property_name}_target_{target_value:.3f}_loglikelihood.png'
        )
        
        # Direct predictions
        direct_preds = plot_direct_property_predictions(
            model, samples, property_index, prop_mean, prop_std,
            target_value, property_name,
            f'{plots_dir}/{property_name}_target_{target_value:.3f}_direct_predictions.png',
            dataset_samples
        )
        direct_predictions_list.append(direct_preds)
        
        # Decode and analyze molecules
        valid_mols, predictions, log_likelihoods = decode_encode_posterior_samples(
            model, samples, property_index, prop_mean, prop_std, target_value
        )
        
        # Select best molecules
        best_mols, best_preds = select_best_molecules(valid_mols, predictions, log_likelihoods)
        selected_predictions_list.append(best_preds)
        
        # Get dataset values for selected molecules
        dataset_values = []
        for mol in best_mols:
            smiles = Chem.MolToSmiles(mol)
            if smiles in set(entire_data['smiles']):
                val = entire_data[entire_data['smiles'] == smiles][property_name].iloc[0]
                dataset_values.append(val)
            else:
                dataset_values.append(None)
        
        # Create molecule visualization grid
        create_molecule_grid(
            best_mols,
            best_preds,
            dataset_values,
            property_name,
            target_value,
            f'{molecules_dir}/{property_name}_target_{target_value:.3f}_molecules.png'
        )
        
        # Rest of the function remains the same...
        plot_selected_property_predictions(
            best_preds, 
            target_value, 
            property_name,
            f'{plots_dir}/{property_name}_target_{target_value:.3f}_selected_predictions.png',
            dataset_samples
        )
        
        df = save_molecule_data(
            best_mols, best_preds, property_name, target_value,
            entire_data, f'{analysis_dir}/{property_name}_target_{target_value:.3f}_molecules.csv'
        )
        
        non_novel = df[df['exists_in_dataset']]
        if len(non_novel) > 0:
            non_novel.to_csv(f'{analysis_dir}/{property_name}_target_{target_value:.3f}_nonnovel.csv')
        
        all_mols.extend(best_mols)
        all_diagnostics.append({'target': target_value, 'mcmc': mcmc})
    
    # Plot combined distributions
    plot_property_kde(
        dataset_samples=dataset_samples,
        target_samples=direct_predictions_list,
        target_values=target_values,
        title=f"Direct Property Distribution - {property_name}",
        label="Direct Predictions",
        property_name=property_name,
        save_path=f'{plots_dir}/{property_name}_combined_direct.png'
    )
    
    plot_property_kde(
        dataset_samples=dataset_samples,
        target_samples=selected_predictions_list,
        target_values=target_values,
        title=f"Selected Molecules Distribution - {property_name}",
        label="Selected Molecules",
        property_name=property_name,
        save_path=f'{plots_dir}/{property_name}_combined_selected.png'
    )
    
    return direct_predictions_list, selected_predictions_list, all_mols, all_diagnostics

def main():
    # Paths and initialization
    vocab_path = '../../data/vocab.txt'
    model_path = '../../joint_training/entire_data_after_error_check/model.best'
    analysis_dir = './point_sampling_results_modelv3/'
    # Load dataset for statistics
    entire_data = pd.read_csv('../../data/qm9_smiles_prop_normalized.csv')

    # Calculate means and standard deviations
    m3 = entire_data['r2'].mean()
    m2 = entire_data['lumo'].mean()
    m1 = entire_data['homo'].mean()

    s3 = entire_data['r2'].std()
    s2 = entire_data['lumo'].std()
    s1 = entire_data['homo'].std()
    
    # Create output directories
    plots_dir = os.path.join(analysis_dir, 'plots')
    summaries_dir = os.path.join(analysis_dir, 'summaries')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    
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
    
    property_to_target_value_map = {
        "homo": [-0.3, -0.27, -0.23, -0.2, -0.17],
        "lumo": [-0.15, -0.07, 0, 0.07, 0.15],
        "r2": [500, 1000, 1500, 2000, 2500]
    }

    property_indices = {
        "homo": 0,
        "lumo": 1,
        "r2": 2
    }

    property_means = {
        "homo": m1,
        "lumo": m2,
        "r2": m3
    }

    property_stds = {
        "homo": s1,
        "lumo": s2,
        "r2": s3
    }
    
    results = {}
    for prop_name, target_values in property_to_target_value_map.items():

        predictions, selected_mols, mols, diagnostics = process_property(
            model=model,
            property_name=prop_name,
            target_values=target_values,
            property_index=property_indices[prop_name],
            prop_mean=property_means[prop_name],
            prop_std=property_stds[prop_name],
            analysis_dir=analysis_dir,
            entire_data=entire_data
        )
        results[prop_name] = {
            'predictions': predictions,
            'selected_mols': selected_mols,
            'mols': mols,
            'diagnostics': diagnostics
        }
    
    return results

if __name__ == "__main__":
    results = main()