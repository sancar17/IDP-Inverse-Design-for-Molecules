import torch
import pandas as pd
from rdkit import Chem
from PIL import Image, ImageDraw, ImageFont
import io
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import numpy as np
from collections import Counter
from fast_jtnn import *
from fast_jtnn.joint_model_v3_3p import JTNNVAE_joint
from fast_jtnn.datautils import tensorize

def create_molecule_grid(mols, predictions, dataset_values, smiles_list, frequencies=None, input_mol=None, input_pred=None, input_real=None, input_smiles=None, n_cols=3):
    """Create a grid of molecule visualizations with predictions and frequencies"""
    n_mols = len(mols) + (1 if input_mol else 0)
    n_rows = (n_mols + n_cols - 1) // n_cols
    
    mol_size = (300, 350)
    img_size = (mol_size[0] * n_cols, mol_size[1] * n_rows + 60)
    
    grid = Image.new('RGB', img_size, (252, 252, 252))
    draw = ImageDraw.Draw(grid)
    
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        title_font = ImageFont.load_default()
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    def draw_molecule(mol, pred, real_val, smiles, pos_x, pos_y, is_input=False, freq=None):
        mol_copy = Chem.Mol(mol)
        try:
            mol_copy.GetConformer()
        except:
            AllChem.Compute2DCoords(mol_copy)
        
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(mol_size[0], mol_size[1]-100)
        drawer.drawOptions().clearBackground = True
        drawer.drawOptions().backgroundColour = (1, 1, 1)
        drawer.drawOptions().bondLineWidth = 2
        
        drawer.DrawMolecule(mol_copy)
        drawer.FinishDrawing()
        
        png_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png_data))
        
        draw.rectangle([pos_x, pos_y - 25, pos_x + mol_size[0], pos_y + mol_size[1]], fill='white')
        draw.rectangle([pos_x, pos_y - 25, pos_x + mol_size[0], pos_y + mol_size[1]], 
                      outline=(200, 200, 200), width=1)
        
        img_x = pos_x + (mol_size[0] - img.width) // 2
        img_y = pos_y + 10
        grid.paste(img, (img_x, img_y))
        
        if is_input:
            title = "Input Structure"
        elif freq is not None:
            title = f"Generated Structure (Freq: {freq:.1f}%)"
        else:
            title = "Generated Structure"
        
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = pos_x + (mol_size[0] - title_width) // 2
        draw.text((title_x, pos_y - 20), title, font=title_font, fill=(50, 50, 50))
        
        max_len = 40
        display_smiles = smiles if len(smiles) <= max_len else smiles[:max_len] + "..."
        smiles_bbox = draw.textbbox((0, 0), display_smiles, font=small_font)
        smiles_width = smiles_bbox[2] - smiles_bbox[0]
        smiles_x = pos_x + (mol_size[0] - smiles_width) // 2
        draw.text((smiles_x, pos_y + mol_size[1] - 135), display_smiles, font=small_font, fill=(100, 100, 100))
        
        if isinstance(pred, (list, tuple, np.ndarray)):
            start_y = pos_y + mol_size[1] - 110
            spacing = 25
            for i, (p, label) in enumerate(zip(pred, ['HOMO', 'LUMO', 'R2'])):
                if isinstance(real_val, (list, tuple, np.ndarray)) and real_val[i] is not None:
                    if is_input:
                        line = f"{label:<5} pred: {p:>8.3f}     ground truth: {real_val[i]:>8.3f}"
                    else:
                        line = f"{label:<5} pred: {p:>8.3f}     real: {real_val[i]:>8.3f}"
                else:
                    line = f"{label:<5} pred: {p:>8.3f}"
                
                line_bbox = draw.textbbox((0, 0), line, font=font)
                line_width = line_bbox[2] - line_bbox[0]
                line_x = pos_x + (mol_size[0] - line_width) // 2
                text_padding = 2
                draw.rectangle([line_x - text_padding, 
                              start_y + i * spacing - text_padding,
                              line_x + line_width + text_padding,
                              start_y + i * spacing + line_bbox[3] - line_bbox[1] + text_padding],
                              fill='white')
                draw.text((line_x, start_y + i * spacing), line, font=font, fill=(50, 50, 50))
    
    current_idx = 0
    if input_mol:
        row = current_idx // n_cols
        col = current_idx % n_cols
        x_offset = col * mol_size[0]
        y_offset = row * mol_size[1] + 40
        draw_molecule(input_mol, input_pred, input_real, input_smiles, x_offset, y_offset, is_input=True)
        current_idx += 1
    
    for idx, (mol, pred, real_val, smiles) in enumerate(zip(mols, predictions, dataset_values, smiles_list)):
        row = (idx + current_idx) // n_cols
        col = (idx + current_idx) % n_cols
        x_offset = col * mol_size[0]
        y_offset = row * mol_size[1] + 40
        freq = frequencies[idx] if frequencies is not None else None
        draw_molecule(mol, pred, real_val, smiles, x_offset, y_offset, freq=freq)
    
    return grid

def analyze_molecule(model, smiles, vocab, entire_data, n_encodings=10, n_decodings=10):
    """Analyze a single molecule with multiple encodings and decodings"""
    device = next(model.parameters()).device
    
    # Calculate normalization parameters
    m3 = entire_data['r2'].mean()
    m2 = entire_data['lumo'].mean()
    m1 = entire_data['homo'].mean()
    s3 = entire_data['r2'].std()
    s2 = entire_data['lumo'].std()
    s1 = entire_data['homo'].std()

    # List to store all generated molecules
    all_generated = []
    all_predictions = []
    all_real_values = []
    
    # Get input molecule predictions
    tree_batch = [MolTree(smiles)]
    _, jtenc_holder, mpn_holder = tensorize(tree_batch, vocab, assm=False)
    input_tree_vecs, _, input_mol_vecs = model.encode(jtenc_holder, mpn_holder)
    
    z_prop = torch.cat([input_mol_vecs[:, :model.z_prop_size//2], 
                       input_tree_vecs[:, :model.z_prop_size//2]], dim=-1)
    input_predictions = model.propNN(z_prop)
    input_pred_norm = input_predictions[:, :3].cpu().detach().numpy()[0]
    
    input_pred = np.array([
        input_pred_norm[0] * s1 + m1,
        input_pred_norm[1] * s2 + m2,
        input_pred_norm[2] * s3 + m3
    ])
    
    # Get real values for input
    input_real = None
    if smiles in set(entire_data['smiles']):
        mol_data = entire_data[entire_data['smiles'] == smiles].iloc[0]
        input_real = [mol_data['homo'], mol_data['lumo'], mol_data['r2']]

    # Perform multiple encodings
    for _ in range(n_encodings):
        tree_vecs, _, mol_vecs = model.encode(jtenc_holder, mpn_holder)
        z_tree, _ = model.rsample(tree_vecs, model.T_mean, model.T_var)
        z_mol, _ = model.rsample(mol_vecs, model.G_mean, model.G_var)
        
        # Perform multiple decodings for each encoding
        for _ in range(n_decodings):
            try:
                decoded_smiles = model.decode(z_tree, z_mol, prob_decode=True)
                if decoded_smiles and Chem.MolFromSmiles(decoded_smiles):
                    # Get predictions
                    pred_norm = model.predict_property_from_smiles([decoded_smiles])
                    pred_norm = pred_norm[:, :3].cpu().detach().numpy()[0]
                    
                    pred = np.array([
                        pred_norm[0] * s1 + m1,
                        pred_norm[1] * s2 + m2,
                        pred_norm[2] * s3 + m3
                    ])
                    
                    # Get real values if available
                    real_val = None
                    if decoded_smiles in set(entire_data['smiles']):
                        mol_data = entire_data[entire_data['smiles'] == decoded_smiles].iloc[0]
                        real_val = [mol_data['homo'], mol_data['lumo'], mol_data['r2']]
                    
                    all_generated.append(decoded_smiles)
                    all_predictions.append(pred)
                    all_real_values.append(real_val)
            except Exception as e:
                print(f"Error in decoding: {e}")
                continue

    # Calculate frequencies
    counter = Counter(all_generated)
    total = len(all_generated)
    
    # Create frequency DataFrame and save to CSV
    freq_df = pd.DataFrame([
        {
            'SMILES': smile,
            'Count': count,
            'Percentage': (count / total) * 100
        }
        for smile, count in counter.items()
    ]).sort_values('Percentage', ascending=False)
    
    freq_df.to_csv('generated_molecules.csv', index=False)
    print(f"\nGenerated molecules statistics saved to 'generated_molecules.csv'")
    print(f"Total unique molecules generated: {len(freq_df)}")
    print("\nTop 10 most frequent molecules:")
    print(freq_df.head(10).to_string(index=False))

    # Get top 5 most frequent molecules
    top_5_smiles = freq_df.head(5)['SMILES'].tolist()
    top_5_freqs = freq_df.head(5)['Percentage'].tolist()
    
    # Get corresponding predictions and real values
    top_5_indices = [all_generated.index(smiles) for smiles in top_5_smiles]
    top_5_predictions = [all_predictions[i] for i in top_5_indices]
    top_5_real_values = [all_real_values[i] for i in top_5_indices]

    return {
        'input_mol': Chem.MolFromSmiles(smiles),
        'input_pred': input_pred,
        'input_real': input_real,
        'input_smiles': smiles,
        'decoded_mols': [Chem.MolFromSmiles(s) for s in top_5_smiles],
        'decoded_predictions': top_5_predictions,
        'decoded_real_values': top_5_real_values,
        'decoded_smiles': top_5_smiles,
        'frequencies': top_5_freqs
    }

def main(smiles, model_path, vocab_path, data_path):
    # Load model and data
    vocab = [x.strip("\r\n ") for x in open(vocab_path)]
    vocab = Vocab(vocab)
    
    hidden_size = 450
    latent_size = 56
    depthT = 20
    depthG = 3
    
    model = JTNNVAE_joint(vocab, hidden_size, latent_size, depthT, depthG, property_weight=1, z_prop_size=14)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    
    entire_data = pd.read_csv(data_path)
    
    # Analyze molecule with multiple encodings/decodings
    results = analyze_molecule(model, smiles, vocab, entire_data)
    
    # Create visualization
    regular_grid = create_molecule_grid(
        results['decoded_mols'],
        results['decoded_predictions'],
        results['decoded_real_values'],
        results['decoded_smiles'],
        results['frequencies'],
        results['input_mol'],
        results['input_pred'],
        results['input_real'],
        results['input_smiles']
    )
    
    # Create safe filename from SMILES
    safe_name = "".join(x for x in smiles if x.isalnum())[:30]
    
    return regular_grid, safe_name

if __name__ == "__main__":
    smiles = "O"  
    vocab_path = '../../data/vocab.txt'
    model_path = '../../joint_training/entire_data_after_error_check/model.best'
    data_path = '../../data/qm9_smiles_prop_normalized.csv'
    
    regular_grid, safe_name = main(smiles, model_path, vocab_path, data_path)
    regular_grid.save(f"v3/encoding_analysis_{safe_name}.png")