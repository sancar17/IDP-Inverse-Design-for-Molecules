import sys
import torch
import torch.nn as nn
import argparse
from fast_jtnn import *
import rdkit
import numpy as np
from fast_jtnn.joint_model_final_version import JTNNVAE_joint

def load_model(vocab, model_path, hidden_size=450, latent_size=56, depthT=20, depthG=3, zprop_size=28):
    vocab = [x.strip("\r\n ") for x in open(vocab)] 
    vocab = Vocab(vocab)

    model = JTNNVAE_joint(vocab, hidden_size, latent_size, depthT, depthG, zprop_size)
    dict_buffer = torch.load(model_path)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()

    torch.manual_seed(0)
    return model

def main_sample(vocab, output_file, latent_file, model_path, nsample, hidden_size=450, latent_size=56, depthT=20, depthG=3, zprop_size=28):

    print('reading vocab')
    vocab = [x.strip("\r\n ") for x in open(vocab)] 
    vocab = Vocab(vocab)

    print('loading model')
    print("new_version")
    model = JTNNVAE_joint(vocab, int(hidden_size), int(latent_size), int(depthT), int(depthG), property_weight=1, z_prop_size=zprop_size)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()

    print('start sampling')
    torch.manual_seed(5)
    smiles_list = []
    latent_points = []
    with open(output_file, 'w') as out_file:
        for i in range(nsample):
            #print(i)
            z_tree = torch.randn(1, model.latent_size).cuda()
            z_mol = torch.randn(1, model.latent_size).cuda()
            smiles = model.decode(z_tree, z_mol, prob_decode=False)
            if smiles is not None:
                try:
                    _ = MolTree(smiles)  # Ensure the molecule is valid
                    smiles_list.append(smiles)
                    out_file.write(smiles + '\n')
                except KeyError:
                    print(f"Generated SMILES not in vocab: {smiles}")
                    continue

    # Encode the SMILES strings to latent space individually
    '''for smiles in smiles_list:
        try:
            mol_vec = model.encode_from_smiles([smiles])
            latent_points.append(mol_vec.data.cpu().numpy())
        except KeyError as e:
            print(f"Error encoding SMILES: {e}, SMILES: {smiles}")
            continue'''

    #latent_points = np.vstack(latent_points)
    #np.save(latent_file, latent_points)

if __name__ == '__main__':
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--nsample', type=int, required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--latent_file', required=True)
    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--latent_size', type=int, default=56)
    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)

    args = parser.parse_args()
    
    main_sample(args.vocab, args.output_file, args.latent_file, args.model, args.nsample, args.hidden_size, args.latent_size, args.depthT, args.depthG)
