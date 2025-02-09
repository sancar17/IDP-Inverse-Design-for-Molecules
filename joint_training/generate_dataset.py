import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch
from fast_jtnn import *
from sklearn.model_selection import train_test_split
from fast_jtnn.mol_tree import main_mol_tree


#read propoerty csv
inp = pd.read_csv('../data/excluded1_qm9_smiles_prop.csv')
inp['INCHI'] = inp['smiles'].apply(lambda x: Chem.MolToInchiKey(Chem.MolFromSmiles(x)))
inp = inp.drop_duplicates(subset=['INCHI'], keep='first')

#save first 10000 unique molecules as csv
#inp = inp[:10000]
unique_csv_path = '../data/excluded1_qm9_smiles_prop.csv'
inp.to_csv(unique_csv_path, index=False)

# Partition the data into 90-10 train-test split
train, test = train_test_split(inp, test_size=0.1, random_state=42)

# Save the train and test data to CSV files
train_csv_path = '../data/excluded1_qm9_smiles_train.csv'
test_csv_path = '../data/excluded1_qm9_smiles_test.csv'

train.to_csv(train_csv_path, index=False)
test.to_csv(test_csv_path, index=False)

# Save only the smiles strings to text files
train_txt_path = '../data/excluded1_qm9_smiles_train.txt'
test_txt_path = '../data/excluded1_qm9_smiles_test.txt'
smiles_txt_path = '../data/excluded1_qm9_smiles.txt'

with open(train_txt_path, 'w') as f:
    for smiles in train['smiles']:
        f.write(smiles + '\n')

with open(test_txt_path, 'w') as f:
    for smiles in test['smiles']:
        f.write(smiles + '\n')

with open(smiles_txt_path, 'w') as f:
    for smiles in inp['smiles']:
        f.write(smiles + '\n')

#normalize properties
average_homo = inp['homo'].mean()
average_lumo = inp['lumo'].mean()
average_r2 = inp['r2'].mean()

std_homo = inp['homo'].std()
std_lumo = inp['lumo'].std()
std_r2 = inp['r2'].std()

print(f'Average HOMO: {average_homo}')
print(f'Average LUMO: {average_lumo}')
print(f'Average R2: {average_r2}')

print(f'std HOMO: {std_homo}')
print(f'std LUMO: {std_lumo}')
print(f'std R2: {std_r2}')

inp['normalized_homo'] = (inp['homo'] - average_homo) / std_homo
inp['normalized_lumo'] = (inp['lumo'] - average_lumo) / std_lumo
inp['normalized_r2'] = (inp['r2'] - average_r2) / std_r2

# Calculate and print normalized mean and standard deviation
normalized_mean_homo = inp['normalized_homo'].mean()
normalized_mean_lumo = inp['normalized_lumo'].mean()
normalized_mean_r2 = inp['normalized_r2'].mean()

normalized_std_homo = inp['normalized_homo'].std()
normalized_std_lumo = inp['normalized_lumo'].std()
normalized_std_r2 = inp['normalized_r2'].std()

print(f'Normalized Average HOMO: {normalized_mean_homo}')
print(f'Normalized Average LUMO: {normalized_mean_lumo}')
print(f'Normalized Average R2: {normalized_mean_r2}')

print(f'Normalized std HOMO: {normalized_std_homo}')
print(f'Normalized std LUMO: {normalized_std_lumo}')
print(f'Normalized std R2: {normalized_std_r2}')

# Save the updated dataframe
inp.to_csv('../data/excluded1_qm9_smiles_prop_normalized.csv', index=False)

#generate vocab
print("generating vocab")
main_mol_tree('../data/excluded1_qm9_smiles.txt', '../data/excluded1_vocab.txt')

'''
10k dataset
Average HOMO: -0.24090058999999925
Average LUMO: 0.009732689999999954
Average R2: 1205.7043235699982
std HOMO: 0.021923836893078502
std LUMO: 0.04636942607722656
std R2: 284.0081214582306
Normalized Average HOMO: -3.4303218598985554e-14
Normalized Average LUMO: 1.0071610212492032e-15
Normalized Average R2: 6.298439547691714e-15
Normalized std HOMO: 1.0000000000000027
Normalized std LUMO: 0.9999999999999986
Normalized std R2: 1.0000000000000022
generating vocab'''

'''
20k dataset
Average HOMO: -0.24106913499999846
Average LUMO: 0.009145149999999932
Average R2: 1205.51403614
std HOMO: 0.022033741446696647
std LUMO: 0.046099723162957094
std R2: 284.27682444382674
Normalized Average HOMO: -6.971864197069522e-14
Normalized Average LUMO: 1.456995635251701e-15
Normalized Average R2: 3.0596913891400845e-16
Normalized std HOMO: 0.9999999999999971
Normalized std LUMO: 0.9999999999999947
Normalized std R2: 1.0000000000000013'''

'''40k dataset
Average HOMO: -0.24116888249999865
Average LUMO: 0.009250892499999969
Average R2: 1207.1385179400013
std HOMO: 0.021998475794350457
std LUMO: 0.04621108203998325
std R2: 285.34158432486856
Normalized Average HOMO: -6.119739714982585e-14
Normalized Average LUMO: 6.668221530503615e-16
Normalized Average R2: -4.526805319482463e-15
Normalized std HOMO: 1.0000000000000018
Normalized std LUMO: 0.9999999999999932
Normalized std R2: 1.0'''

'''entire dataset
Average HOMO: -0.24132133937470535
Average LUMO: 0.009734840325295804
Average R2: 1206.6448024355188
std HOMO: 0.02182456631867619
std LUMO: 0.04639765091106505
std R2: 283.55181046343085
Normalized Average HOMO: 1.7018495201372687e-13
Normalized Average LUMO: 1.1641855226453211e-15
Normalized Average R2: 4.387985808935869e-14
Normalized std HOMO: 1.0000000000000282
Normalized std LUMO: 0.9999999999999238
Normalized std R2: 1.0000000000000027'''

'''excluded1 dataset
Average HOMO: -0.24056326327491329
Average LUMO: 0.009300360632852422
Average R2: 1130.1039665774758
std HOMO: 0.021905646482926117
std LUMO: 0.05046597136933291
std R2: 275.67127500578374
Normalized Average HOMO: 2.3096458478929826e-13
Normalized Average LUMO: 1.3132774257281715e-15
Normalized Average R2: 7.49942188644959e-15
Normalized std HOMO: 0.9999999999999994
Normalized std LUMO: 1.000000000000025
Normalized std R2: 0.9999999999999964
'''