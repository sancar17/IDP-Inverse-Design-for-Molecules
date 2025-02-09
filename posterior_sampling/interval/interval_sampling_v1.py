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
from fast_jtnn.joint_model_v1_3p import JTNNVAE_joint
from rdkit.Chem import Draw, AllChem
from PIL import Image, ImageDraw, ImageFont
import io

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def analyze_dataset(data_path):
    """Analyze dataset and determine meaningful intervals for each property"""
    df = pd.read_csv(data_path)
    intervals = {}
    
    properties = ['homo', 'lumo', 'r2']
    for prop in properties:
        values = df[prop].values
        q1, q2, q3 = np.percentile(values, [25, 50, 75])
        iqr = q3 - q1
        
        low = q1 - 0.5 * iqr
        high = q3 + 0.5 * iqr
        
        intervals[prop] = {
            'intervals': [(low, q1), (q1, q3), (q3, high)],
            'stats': {
                'mean': np.mean(values),
                'std': np.std(values),
                'percentiles': [q1, q2, q3]
            }
        }
        
        '''for i, (start, end) in enumerate(intervals[prop]['intervals']):
            mask = (df[prop] >= start) & (df[prop] <= end)
            subset = df[mask]
            subset.to_csv(f'{prop}_interval_{i+1}.csv', index=False)'''
    
    return intervals

def plot_selected_interval_predictions(best_preds, target_interval, property_name, save_path, dataset_samples):
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
    
    plt.axvspan(target_interval[0], target_interval[1], alpha=0.2, color='gray',
                label=f'Target [{target_interval[0]:.2f}, {target_interval[1]:.2f}]')
    plt.xlabel(property_name)
    plt.ylabel('Density')
    plt.title(f'{property_name} Selected Molecules Distribution')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_combined_interval_predictions(dataset_samples, selected_predictions_list, target_intervals, 
                                    property_name, save_path):
    """Plot combined distributions for all intervals"""
    plt.figure(figsize=(12, 6))
    
    # Dataset distribution
    kde_dataset = gaussian_kde(dataset_samples)
    x_dataset = np.linspace(min(dataset_samples), max(dataset_samples), 2048)
    plt.plot(x_dataset, kde_dataset(x_dataset), 'r-', linewidth=2, label='Dataset')
    
    # Predictions for each interval
    cmap = plt.cm.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(target_intervals)))
    
    for i, (predictions, interval) in enumerate(zip(selected_predictions_list, target_intervals)):
        kde = gaussian_kde(predictions)
        x_pred = np.linspace(min(predictions), max(predictions), 2048)
        plt.plot(x_pred, kde(x_pred), '--', color=colors[i], linewidth=2,
                label=f'Interval [{interval[0]:.2f}, {interval[1]:.2f}]')
        plt.axvspan(interval[0], interval[1], alpha=0.1, color=colors[i])
    
    plt.xlabel(property_name)
    plt.ylabel('Density')
    plt.title(f'{property_name} Selected Molecules Distribution - All Intervals')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def sample_posterior_interval(model, target_interval, property_index, dataset_mean, dataset_std):
    """Sample from posterior for a target interval"""
    interval_start = (torch.tensor([target_interval[0]], dtype=torch.float32).cuda() - dataset_mean) / dataset_std
    interval_end = (torch.tensor([target_interval[1]], dtype=torch.float32).cuda() - dataset_mean) / dataset_std
    all_trajectories = []
    current_trajectory = []
    
    def model_fn():
        z_prop = pyro.sample(
            "z_prop",
            dist.Normal(torch.zeros(56, dtype=torch.float32).cuda(), 
                       torch.ones(56, dtype=torch.float32).cuda())
        ).unsqueeze(0)
        
        out = model.propNN(z_prop)
        y_mu, y_sigma_log = out[:, property_index], out[:, property_index + 3]
        y_sigma = torch.exp(y_sigma_log)
        
        # Sample from uniform distribution over interval
        y = pyro.sample("y", dist.Uniform(interval_start, interval_end))
        
        # Calculate probability of being in interval under the model
        normal = dist.Normal(y_mu, y_sigma)
        prob = normal.cdf(interval_end) - normal.cdf(interval_start)
        current_trajectory.append(torch.log(prob).item())
        
        return y
    
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
    

def plot_interval_predictions(model, samples, property_index, prop_mean, prop_std, 
                            target_interval, property_name, save_path, dataset_samples):
    with torch.no_grad():
        out = model.propNN(samples)
        y_pred = out[:, property_index] * prop_std + prop_mean
        y_pred = y_pred.cpu().numpy()

    plt.figure(figsize=(10, 6))
    kde_dataset = gaussian_kde(dataset_samples)
    x_dataset = np.linspace(min(dataset_samples), max(dataset_samples), 2048)
    plt.plot(x_dataset, kde_dataset(x_dataset), 'r-', label='Dataset')
    
    kde_pred = gaussian_kde(y_pred)
    x_pred = np.linspace(min(y_pred), max(y_pred), 2048)
    plt.plot(x_pred, kde_pred(x_pred), 'b-', label='Predicted')
    
    plt.axvspan(target_interval[0], target_interval[1], alpha=0.2, color='gray',
                label=f'Target [{target_interval[0]:.2f}, {target_interval[1]:.2f}]')
    
    plt.xlabel(property_name)
    plt.ylabel('Density')
    plt.title(f'{property_name} Predictions Distribution')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    return y_pred

def create_molecule_grid(mols, predictions, dataset_values, property_name, target_interval, save_path, n_cols=5):
    n_mols = len(mols)
    n_rows = (n_mols + n_cols - 1) // n_cols
    mol_size = (300, 300)
    img_size = (mol_size[0] * n_cols, mol_size[1] * n_rows + 40)
    
    grid = Image.new('RGB', img_size, 'white')
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    title = f"{property_name} - Target: [{target_interval[0]:.2f}, {target_interval[1]:.2f}]"
    draw.text((10, 10), title, font=font, fill='black')
    
    for idx, (mol, pred, real_val) in enumerate(zip(mols, predictions, dataset_values)):
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
        
        draw.rectangle([x_offset, y_offset, x_offset + mol_size[0], y_offset + 25], fill='white')
        draw.rectangle([x_offset, y_offset + mol_size[1] - 25, x_offset + mol_size[0], 
                      y_offset + mol_size[1]], fill='white')
        
        grid.paste(img, (x_offset, y_offset))
        
        # Color code prediction text based on interval
        if target_interval[0] <= pred <= target_interval[1]:
            pred_color = 'green'
        else:
            pred_color = 'red'
        
        pred_text = f"Pred: {pred:.3f}"
        draw.text((x_offset + 10, y_offset + 5), pred_text, font=font, fill=pred_color)
        
        if real_val is not None:
            val_text = f"Real: {real_val:.3f}"
        else:
            val_text = "Real: ?"
        draw.text((x_offset + 10, y_offset + mol_size[1] - 20), val_text, font=font, fill='black')
        
        draw.rectangle([x_offset, y_offset, x_offset + mol_size[0], 
                       y_offset + mol_size[1]], outline='black', width=1)
    
    grid.save(save_path)

def decode_encode_posterior_samples(model, posterior_samples, property_index, prop_mean, prop_std):
    valid_mols = []
    predictions = []
    smiles_set = set()
    
    with torch.no_grad():
        out = model.propNN(posterior_samples)
        y_pred = out[:, property_index] * prop_std + prop_mean
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
                            predictions.append(y_pred[i].item())
                    except:
                        continue
    
    return valid_mols, predictions

def select_best_molecules_interval(valid_mols, predictions, target_interval, k=20):
    interval_scores = []
    for pred in predictions:
        if target_interval[0] <= pred <= target_interval[1]:
            center = (target_interval[0] + target_interval[1]) / 2
            score = -abs(pred - center)
        else:
            score = -min(abs(pred - target_interval[0]), 
                        abs(pred - target_interval[1]))
        interval_scores.append(score)
    
    indices = np.argsort(interval_scores)[-k:]
    return [valid_mols[i] for i in indices], [predictions[i] for i in indices]

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

def process_property_interval(model, property_name, target_intervals, property_index, prop_mean, prop_std, analysis_dir, entire_data):
    plots_dir = os.path.join(analysis_dir, 'plots')
    molecules_dir = os.path.join(analysis_dir, 'molecule_visualizations')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(molecules_dir, exist_ok=True)
    
    results = []
    dataset_samples = entire_data[property_name].values
    
    for target_interval in target_intervals:
        print(f"Processing {property_name} for interval: [{target_interval[0]:.2f}, {target_interval[1]:.2f}]")
        
        samples, mcmc, log_likelihoods = sample_posterior_interval(
            model, target_interval, property_index, prop_mean, prop_std
        )
        
        plot_log_likelihood(
            log_likelihoods,
            f'{plots_dir}/{property_name}_interval_{target_interval[0]:.2f}_{target_interval[1]:.2f}_loglikelihood.png'
        )
        
        y_pred = plot_interval_predictions(
            model, samples, property_index, prop_mean, prop_std,
            target_interval, property_name,
            f'{plots_dir}/{property_name}_interval_{target_interval[0]:.2f}_{target_interval[1]:.2f}_predictions.png',
            dataset_samples
        )
        
        valid_mols, predictions = decode_encode_posterior_samples(
            model, samples, property_index, prop_mean, prop_std
        )
        
        best_mols, best_preds = select_best_molecules_interval(
            valid_mols, predictions, target_interval
        )
        
        dataset_values = []
        for mol in best_mols:
            smiles = Chem.MolToSmiles(mol)
            if smiles in set(entire_data['smiles']):
                val = entire_data[entire_data['smiles'] == smiles][property_name].iloc[0]
                dataset_values.append(val)
            else:
                dataset_values.append(None)
        
        create_molecule_grid(
            best_mols,
            best_preds,
            dataset_values,
            property_name,
            target_interval,
            f'{molecules_dir}/{property_name}_interval_{target_interval[0]:.2f}_{target_interval[1]:.2f}_molecules.png'
        )
        
        results.append({
            'interval': target_interval,
            'predictions': y_pred,
            'best_molecules': best_mols,
            'best_predictions': best_preds
        })

        plot_selected_interval_predictions(
        best_preds,
        target_interval,
        property_name,
        f'{plots_dir}/{property_name}_interval_{target_interval[0]:.2f}_{target_interval[1]:.2f}_selected.png',
        dataset_samples
    )

    # At the end of the function, after processing all intervals:
    plot_combined_interval_predictions(
        dataset_samples,
        [result['best_predictions'] for result in results],
        [result['interval'] for result in results],
        property_name,
        f'{plots_dir}/{property_name}_combined_intervals.png'
    )
    
    return results

def main():
    vocab_path = '../../data/vocab.txt'
    model_path = '../../joint_training/v1_3p_entire_data/model.best'
    analysis_dir = './interval_sampling_results_v1/'
    data_path = '../../data/qm9_smiles_prop_normalized.csv'
    
    entire_data = pd.read_csv(data_path)
    intervals = analyze_dataset(data_path)
    
    m1, m2, m3 = entire_data['homo'].mean(), entire_data['lumo'].mean(), entire_data['r2'].mean()
    s1, s2, s3 = entire_data['homo'].std(), entire_data['lumo'].std(), entire_data['r2'].std()
    
    os.makedirs(analysis_dir, exist_ok=True)
    
    vocab = [x.strip("\r\n ") for x in open(vocab_path)]
    vocab = Vocab(vocab)
    
    model = JTNNVAE_joint(vocab, 450, 56, 20, 3, property_weight=1)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    
    property_indices = {"homo": 0, "lumo": 1, "r2": 2}
    property_means = {"homo": m1, "lumo": m2, "r2": m3}
    property_stds = {"homo": s1, "lumo": s2, "r2": s3}
    
    results = {}
    for prop_name in property_indices:
        results[prop_name] = process_property_interval(
            model=model,
            property_name=prop_name,
            target_intervals=intervals[prop_name]['intervals'],
            property_index=property_indices[prop_name],
            prop_mean=property_means[prop_name],
            prop_std=property_stds[prop_name],
            analysis_dir=analysis_dir,
            entire_data=entire_data
        )
    
    return results, intervals

if __name__ == "__main__":
    results, intervals = main()