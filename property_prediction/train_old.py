import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from optparse import OptionParser
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import wandb
from fast_jtnn import *

class PropertyPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PropertyPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_predictor(model, train_loader, test_loader, criterion, optimizer, epochs, early_stopping_patience):
    best_test_loss = np.inf
    no_improvement_epochs = 0
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
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Calculate individual losses for logging
            homo_loss = criterion(outputs[:, 0], y[:, 0]).item()
            lumo_loss = criterion(outputs[:, 1], y[:, 1]).item()
            r2_loss = criterion(outputs[:, 2], y[:, 2]).item()
            epoch_homo_loss += homo_loss
            epoch_lumo_loss += lumo_loss
            epoch_r2_loss += r2_loss

        # Logging training loss to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_total_loss": epoch_loss / len(train_loader),
            "train_homo_loss": epoch_homo_loss / len(train_loader),
            "train_lumo_loss": epoch_lumo_loss / len(train_loader),
            "train_r2_loss": epoch_r2_loss / len(train_loader)
        })
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss/len(train_loader)}")

        # Evaluation on test set
        model.eval()
        test_loss = 0
        test_homo_loss = 0
        test_lumo_loss = 0
        test_r2_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.cuda(), y.cuda()
                outputs = model(x)
                loss = criterion(outputs, y)
                test_loss += loss.item()

                # Calculate individual losses for logging
                homo_loss = criterion(outputs[:, 0], y[:, 0]).item()
                lumo_loss = criterion(outputs[:, 1], y[:, 1]).item()
                r2_loss = criterion(outputs[:, 2], y[:, 2]).item()
                test_homo_loss += homo_loss
                test_lumo_loss += lumo_loss
                test_r2_loss += r2_loss

        # Logging test loss to wandb
        wandb.log({
            "epoch": epoch + 1,
            "test_total_loss": test_loss / len(test_loader),
            "test_homo_loss": test_homo_loss / len(test_loader),
            "test_lumo_loss": test_lumo_loss / len(test_loader),
            "test_r2_loss": test_r2_loss / len(test_loader)
        })
        print(f"Epoch {epoch+1}/{epochs}, Test Loss: {test_loss/len(test_loader)}")

        if(test_loss / len(test_loader)< best_test_loss):
            best_test_loss = test_loss / len(test_loader)
            no_improvement_epochs = 0

        else:
            no_improvement_epochs += 1

        if(no_improvement_epochs == early_stopping_patience):
            return

def main():
    parser = OptionParser()
    parser.add_option("-s", "--smiles", dest="smiles_path")
    parser.add_option("-c", "--csv", dest="csv_path")
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
    options, args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="property_predictor")
    wandb.config.update(options)

    # Load SMILES and data
    with open(options.smiles_path, 'r') as f:
        smiles_list = [line.strip() for line in f]
    
    data = pd.read_csv(options.csv_path)
    
    # Ensure unique SMILES
    data = data.drop_duplicates(subset='smiles')

    # Reindex data based on SMILES from the text file
    data = data.set_index('smiles').reindex(smiles_list).dropna().reset_index()
    smiles = data['smiles'].tolist()
    properties = data[['homo', 'lumo', 'r2']].values

    # Load vocab and model
    vocab = Vocab([x.strip("\r\n ") for x in open(options.vocab_path)])
    model = JTNNVAE(vocab, options.hidden_size, options.latent_size, options.depthT, options.depthG)
    model.load_state_dict(torch.load(options.model_path))
    model = model.cuda()

    # Encode SMILES to latent space
    latent_points = []
    batch_size = options.batch_size
    for i in range(0, len(smiles), batch_size):
        batch = smiles[i:i+batch_size]
        mol_vec = model.encode_from_smiles(batch)
        latent_points.append(mol_vec.data.cpu().numpy())
    latent_points = np.vstack(latent_points)

    # Prepare dataset
    X_train, X_test, y_train, y_test = train_test_split(latent_points, properties, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())

    train_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=False)

    # Initialize property predictor model
    input_size = 56 # Adjusted for correct latent size
    hidden_size = 256
    output_size = 3
    predictor = PropertyPredictor(input_size, hidden_size, output_size).cuda()

    # Train property predictor
    criterion = nn.MSELoss()
    optimizer = optim.Adam(predictor.parameters(), lr=options.learning_rate)
    train_predictor(predictor, train_loader, test_loader, criterion, optimizer, options.epochs, options.early_stopping_patience)

    # Save the trained model
    torch.save(predictor.state_dict(), options.output_path + '/property_predictor.pth')

if __name__ == "__main__":
    main()
