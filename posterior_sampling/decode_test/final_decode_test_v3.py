import torch
import pandas as pd
from rdkit import Chem
from collections import Counter
from fast_jtnn import *
from fast_jtnn.joint_model_v3_3p import JTNNVAE_joint
from fast_jtnn.datautils import tensorize
import rdkit.RDLogger as rdlog
import warnings

# Suppress RDKit logging
rdlog.DisableLog('rdApp.*')

def get_inchi_key(smiles):
    """Generate InChI Key from SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol)
    except:
        return None

def analyze_molecule(model, smiles, vocab, entire_data, n_encodings=10, n_decodings=10):
    """Analyze a single molecule with multiple encodings and decodings"""
    device = next(model.parameters()).device
    
    # Get input molecule InChI Key
    input_inchi = get_inchi_key(smiles)
    if input_inchi is None:
        return None
    
    # Perform multiple encodings and decodings
    all_decoded_smiles = []
    
    try:
        tree_batch = [MolTree(smiles)]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, vocab, assm=False)
        
        for _ in range(n_encodings):
            tree_vecs, _, mol_vecs = model.encode(jtenc_holder, mpn_holder)
            z_tree, _ = model.rsample(tree_vecs, model.T_mean, model.T_var)
            z_mol, _ = model.rsample(mol_vecs, model.G_mean, model.G_var)
            
            for _ in range(n_decodings):
                try:
                    decoded_smiles = model.decode(z_tree, z_mol, prob_decode=True)
                    if decoded_smiles and Chem.MolFromSmiles(decoded_smiles):
                        all_decoded_smiles.append(decoded_smiles)
                except Exception as e:
                    #print(f"Error in decoding: {e}")
                    continue
    except Exception as e:
        print(f"Error processing molecule {smiles}: {e}")
        return None

    # Calculate frequencies and get reconstruction statistics
    counter = Counter(all_decoded_smiles)
    total_generations = len(all_decoded_smiles)
    
    if total_generations == 0:
        return None
        
    # Find most frequent reconstruction
    most_frequent_smiles = counter.most_common(1)[0][0]
    most_frequent_inchi = get_inchi_key(most_frequent_smiles)
    
    # Calculate reconstruction statistics
    reconstruction_count = sum(1 for s in all_decoded_smiles if get_inchi_key(s) == input_inchi)
    reconstruction_percentage = (reconstruction_count / total_generations) * 100 if total_generations > 0 else 0
    
    return {
        'smiles': smiles,
        'inchi_key': input_inchi,
        'most_frequently_reconstructed': (input_inchi == most_frequent_inchi),
        'reconstruction_percentage': reconstruction_percentage
    }

def main(input_file, model_path, vocab_path, data_path, output_file='reconstruction_analysis_v3.csv'):
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
    
    # Read input SMILES
    with open(input_file, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    # Process each molecule
    results = []
    for i, smiles in enumerate(smiles_list, 1):
        print(f"\rProcessing molecule {i}/{len(smiles_list)}", end='')
        result = analyze_molecule(model, smiles, vocab, entire_data)
        if result:
            results.append(result)
    print("\nAnalysis complete!")
    
    # Create and save results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    # Calculate and print summary statistics
    total_molecules = len(results)
    reconstructed_count = results_df['most_frequently_reconstructed'].sum()
    reconstruction_percentage = (reconstructed_count / total_molecules * 100) if total_molecules > 0 else 0
    avg_reconstruction_percentage = results_df['reconstruction_percentage'].mean()
    
    print(f"\nSummary Statistics:")
    print(f"Total molecules processed: {total_molecules}")
    print(f"Molecules successfully reconstructed as most frequent output: {reconstructed_count} ({reconstruction_percentage:.2f}%)")
    print(f"Average reconstruction percentage across all molecules: {avg_reconstruction_percentage:.2f}%")

if __name__ == "__main__":
    input_file = "../..//data/qm9_smiles.txt"  # File containing SMILES strings, one per line
    vocab_path = '../../data/vocab.txt'
    model_path = '../../joint_training/entire_data_after_error_check/model.best'
    data_path = '../../data/qm9_smiles_prop_normalized.csv'
    
    main(input_file, model_path, vocab_path, data_path)