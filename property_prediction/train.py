import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from optparse import OptionParser
from torch.utils.data import DataLoader, TensorDataset
import wandb
from fast_jtnn import *

class PropertyPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PropertyPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 67),
            nn.BatchNorm1d(67),
            nn.PReLU(),
            nn.Dropout(0.15),
            nn.Linear(67, 67),
            nn.BatchNorm1d(67),
            nn.PReLU(),
            nn.Dropout(0.15),
            nn.Linear(67, 6)  # 3 properties with mean and std for each
        )

    def forward(self, x):
        return self.model(x)

def nll_gaussian(y_pred, y_true):
    means = y_pred[:, :3]
    log_stds = y_pred[:, 3:]
    stds = torch.exp(log_stds)
    loss = ((means - y_true) ** 2) / (2 * stds ** 2) + log_stds
    return loss.mean(), loss[:, 0].mean(), loss[:, 1].mean(), loss[:, 2].mean()  # Return losses for individual properties

def train_predictor(model, train_loader, test_loader, optimizer, epochs, early_stopping_patience):
    best_test_loss = np.inf
    no_improvement_epochs = 0
    criterion = nll_gaussian

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_homo_loss = 0
        epoch_lumo_loss = 0
        epoch_r2_loss = 0

        for batch in train_loader:
            x, y = batch
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            #print(y)
            
            outputs = model(x)

            #print(outputs)
            #exit()

            loss, homo_loss, lumo_loss, r2_loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_homo_loss += homo_loss.item()
            epoch_lumo_loss += lumo_loss.item()
            epoch_r2_loss += r2_loss.item()

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss / len(train_loader),
            "train_homo_loss": epoch_homo_loss / len(train_loader),
            "train_lumo_loss": epoch_lumo_loss / len(train_loader),
            "train_r2_loss": epoch_r2_loss / len(train_loader)
        })
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss/len(train_loader)}")

        model.eval()
        test_loss = 0
        test_homo_loss = 0
        test_lumo_loss = 0
        test_r2_loss = 0
        mse_loss = nn.MSELoss()
        test_mse = 0
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.cuda(), y.cuda()
                outputs = model(x)
                loss, homo_loss, lumo_loss, r2_loss = criterion(outputs, y)
                test_loss += loss.item()
                test_homo_loss += homo_loss.item()
                test_lumo_loss += lumo_loss.item()
                test_r2_loss += r2_loss.item()

                # Extract predicted means and stds
                means = outputs[:, :3]
                stds = torch.exp(outputs[:, 3:])
                
                # Compute MSE for the means
                mse = mse_loss(means, y)
                test_mse += mse.item()  

        wandb.log({
            "epoch": epoch + 1,
            "test_loss": test_loss / len(test_loader),
            "test_homo_loss": test_homo_loss / len(test_loader),
            "test_lumo_loss": test_lumo_loss / len(test_loader),
            "test_r2_loss": test_r2_loss / len(test_loader),
            "test_mse": test_mse / len(test_loader)
        })
        print(f"Epoch {epoch+1}/{epochs}, Test Loss: {test_loss/len(test_loader)}, Test MSE: {test_mse/len(test_loader)}")

        if test_loss / len(test_loader) < best_test_loss:
            best_test_loss = test_loss / len(test_loader)
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs == early_stopping_patience:
            break

def main():
    parser = OptionParser()
    parser.add_option("-v", "--vocab", dest="vocab_path")
    parser.add_option("-m", "--model", dest="model_path")
    parser.add_option("-p", "--early_stopping_patience", dest="early_stopping_patience", default='10')
    parser.add_option("-o", "--output", dest="output_path", default='./')
    parser.add_option("-w", "--hidden", dest="hidden_size", default=450, type="int")
    parser.add_option("-l", "--latent", dest="latent_size", default=56, type="int")
    parser.add_option("-t", "--depthT", dest="depthT", default=20, type="int")
    parser.add_option("-g", "--depthG", dest="depthG", default=3, type="int")
    parser.add_option("-e", "--epochs", dest="epochs", default=200, type="int")
    parser.add_option("-b", "--batch_size", dest="batch_size", default=32, type="int")
    parser.add_option("-r", "--learning_rate", dest="learning_rate", default=0.001, type="float")
    parser.add_option("--train_data", dest="train_data_path")
    parser.add_option("--test_data", dest="test_data_path")
    options, args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="property_predictor")
    wandb.config.update(options)

    # Load preprocessed train and test data
    print("Reading CSV files")
    train_data = pd.read_csv(options.train_data_path)
    test_data = pd.read_csv(options.test_data_path)

    # Ensure numeric data and split features and targets
    print("Ensuring numeric data and splitting features")
    train_data = train_data.apply(pd.to_numeric, errors='coerce')
    test_data = test_data.apply(pd.to_numeric, errors='coerce')

    X_train = train_data.iloc[:, 0].values
    y_train = train_data.iloc[:, 1:4].values
    X_test = test_data.iloc[:, 0].values
    y_test = test_data.iloc[:, 1:4].values

    # Normalize the property values together
    print("Normalizing")
    y_all = np.concatenate([y_train, y_test], axis=0)
    y_mean = y_all.mean(axis=0)
    y_std = y_all.std(axis=0)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    print("loading vae")
    # Load vocab and model
    vocab = Vocab([x.strip("\r\n ") for x in open(options.vocab_path)])
    vae_model = JTNNVAE(vocab, options.hidden_size, options.latent_size, options.depthT, options.depthG)
    vae_model.load_state_dict(torch.load(options.model_path))
    vae_model = vae_model.cuda()

    # Encode SMILES to latent space
    def encode_smiles(smiles_list, model, batch_size):
        latent_points = []
        for i in range(0, len(smiles_list), batch_size):
            print(i, '/', len(smiles_list))
            batch = smiles_list[i:i+batch_size]
            mol_vec = model.encode_from_smiles(batch)
            latent_points.append(mol_vec.data.cpu().numpy())
        return np.vstack(latent_points)

    print("reading smiles")
    # Read and encode SMILES from train and test sets
    with open(options.train_data_path.replace('.csv', '.txt'), 'r') as f:
        train_smiles_list = [line.strip() for line in f]
    with open(options.test_data_path.replace('.csv', '.txt'), 'r') as f:
        test_smiles_list = [line.strip() for line in f]

    print("encoding smiles train")
    X_train = encode_smiles(train_smiles_list, vae_model, options.batch_size)
    print("encoding smiles test")
    X_test = encode_smiles(test_smiles_list, vae_model, options.batch_size)

    # Prepare datasets
    print("Preparing datasets")
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())

    train_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=False)

    # Initialize property predictor model
    print("Creating model")
    input_size = options.latent_size  # Adjusted for correct latent size
    hidden_size = 256
    predictor = PropertyPredictor(input_size, hidden_size).cuda()

    # Train property predictor
    print("Starting training")
    optimizer = optim.Adam(predictor.parameters(), lr=options.learning_rate)
    train_predictor(predictor, train_loader, test_loader, optimizer, options.epochs, options.early_stopping_patience)

    # Save the trained model and normalization parameters
    torch.save({
        'model_state_dict': predictor.state_dict(),
        'y_mean': y_mean,
        'y_std': y_std
    }, options.output_path + '/property_predictor.pth')

if __name__ == "__main__":
    main()
