import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
import wandb

import math, random, sys
import numpy as np
import argparse
from collections import deque
import pickle as pickle

from fast_jtnn import *
import rdkit
from tqdm import tqdm
import os
import logging
from fast_jtnn.datautils import MolTreeFolderJoint
import glob
import json
from fast_jtnn.joint_model_v3_3p import JTNNVAE_joint

def get_gpu_memory_usage():
    return torch.cuda.memory_allocated() / 1024**2, torch.cuda.max_memory_allocated() / 1024**2


def save_args_as_config(args, config_path):
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

def remove_old_checkpoints(save_dir, keep_best=True):
    # Get list of all checkpoint files
    checkpoint_files = glob.glob(os.path.join(save_dir, "model.epoch-*.pth"))

    # Sort files by their creation time (oldest first)
    checkpoint_files.sort(key=os.path.getmtime)

    # Remove all but the latest and the best checkpoint
    for f in checkpoint_files[:-1]:  # Keep the latest
        if not keep_best or "best" not in f:
            os.remove(f)

class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

def main_vae_train(train,
                   test,
                   vocab,
                   save_dir,
                   csv_file,
                   wandb_flag,
                   wandb_name,
                   load_epoch=0,
                   hidden_size=450,
                   batch_size=32,
                   latent_size=56,
                   depthT=20,
                   depthG=3,
                   lr=1e-3,
                   clip_norm=50.0,
                   beta=0.0,
                   step_beta=0.002,
                   max_beta=1.0,
                   warmup=40000,
                   epoch=20,
                   anneal_rate=0.9,
                   anneal_iter=40000, 
                   kl_anneal_iter=2000,
                   print_iter=50, 
                   save_iter=5000,
                   early_stopping_patience=20,
                   property_weight=1,
                   z_prop_size=4,
                   config_file=None,):

    # Initialize Weights & Biases (wandb) for experiment tracking if enabled
    if wandb_flag:
        wandb.init(project="entire data v3", name=wandb_name)
        if config_file:
            with open(config_file, 'r') as f:
                config = json.load(f)
            wandb.config.update(config)

    # Load vocabulary
    vocab = [x.strip("\r\n ") for x in open(vocab)] 
    vocab = Vocab(vocab)

    # Initialize model
    model = JTNNVAE_joint(vocab, int(hidden_size), int(latent_size), int(depthT), int(depthG), property_weight, z_prop_size).cuda()
    logging.info(model)

    # Initialize model parameters
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    
    # Load model if resuming from a checkpoint
    if load_epoch > 0:
        model.load_state_dict(torch.load(save_dir + "/model.epoch-" + str(load_epoch)))

    logging.info("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, anneal_rate)

    # Functions to compute parameter and gradient norms
    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    # Main training loop
    total_step = load_epoch * len(train)
    beta = beta
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    kl_active = False
    
    for epoch_num in range(epoch):
        logging.info("Epoch: %d", epoch_num + 1)

        # Initialize epoch metrics for training
        train_meters = np.zeros(4)  # [KL, Word Accuracy, Topo Accuracy, Assembly Accuracy]
        train_loss_total = 0
        train_batches = 0
        train_prop_loss_total = 0
        train_homo_loss_total = 0
        train_lumo_loss_total = 0
        train_r2_loss_total = 0

        # Training
        model.train()
        print("train loader")
        loader = MolTreeFolderJoint(train, vocab, batch_size, csv_file)
        print("train loader done")

        for batch in loader:
            total_step += 1
            model.zero_grad()
            
            if batch is None:
                continue
            
            if(train_batches % 20 == 0):
                print("Epoch:", epoch_num, " Train batch:", train_batches)
                current_mem, max_mem = get_gpu_memory_usage()
                print(f"Current GPU memory: {current_mem:.2f} MB, Max GPU memory: {max_mem:.2f} MB")

            tree_batch, jtenc_holder, mpn_holder, jtmpn_holder, homos, lumos, r2s = batch
            if tree_batch is None:
                #skipped_batches += 1
                continue
                
            # Forward pass
            # model returns total_loss, kl_div.item(), word_acc, topo_acc, assm_acc, prop_loss.item(), prop_losses[0], prop_losses[1], prop_losses[2]
            loss, kl_div, wacc, tacc, sacc, prop_loss, homo_loss, lumo_loss, r2_loss = model(batch, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            # Update metrics
            train_meters += np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])
            train_loss_total += loss.item()
            train_prop_loss_total += prop_loss
            train_batches += 1
            train_homo_loss_total += homo_loss
            train_lumo_loss_total += lumo_loss
            train_r2_loss_total += r2_loss

        # Update learning rate scheduler
        if total_step % anneal_iter == 0:
            scheduler.step()

        # Update KL beta
        if total_step >= warmup and total_step % kl_anneal_iter == 0:
            beta = min(max_beta, beta + step_beta)
            if not kl_active and beta > 0 and total_step > warmup:
                kl_active = True
                logging.info("KL divergence activated at epoch %d", epoch_num + 1)
                best_val_loss = float('inf')

        # Calculate average training metrics
        avg_train_loss = train_loss_total / train_batches
        avg_train_meters = train_meters / train_batches
        avg_train_prop_loss = train_prop_loss_total / train_batches
        avg_train_homo_loss = train_homo_loss_total / train_batches
        avg_train_lumo_loss = train_lumo_loss_total / train_batches
        avg_train_r2_loss = train_r2_loss_total / train_batches

        # Validation
        model.eval()
        val_loss_total = 0
        val_meters = np.zeros(4)
        val_batches = 0
        val_prop_loss_total = 0
        val_homo_loss_total = 0
        val_lumo_loss_total = 0
        val_r2_loss_total = 0

        print("val loader")
        loader = MolTreeFolderJoint(test, vocab, batch_size, csv_file)
        print("loader done")

        with torch.no_grad():
            for batch in loader:
                try:
                    if batch is None:
                        continue

                    tree_batch, jtenc_holder, mpn_holder, jtmpn_holder, homos, lumos, r2s = batch
                    if tree_batch is None:
                        #skipped_batches += 1
                        continue
                    
                    if(val_batches % 20 == 0):
                        print("Epoch:", epoch_num, " Validation batch:", val_batches)
                        current_mem, max_mem = get_gpu_memory_usage()
                        print(f"Current GPU memory: {current_mem:.2f} MB, Max GPU memory: {max_mem:.2f} MB")

                    if val_batches % 5 == 0:
                        torch.cuda.empty_cache()
                
                    loss, kl_div, wacc, tacc, sacc, prop_loss, homo_loss, lumo_loss, r2_loss = model(batch, beta)
                    val_loss_total += loss.item()
                    val_meters += np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])
                    val_prop_loss_total += prop_loss
                    val_batches += 1
                    val_homo_loss_total += homo_loss
                    val_lumo_loss_total += lumo_loss
                    val_r2_loss_total += r2_loss

                except Exception as e:
                    logging.error(e)
                    sys.exit()
                    continue

        torch.cuda.empty_cache()

        # Calculate average validation metrics
        avg_val_loss = val_loss_total / val_batches
        avg_val_meters = val_meters / val_batches
        avg_val_prop_loss = val_prop_loss_total / val_batches

        avg_val_homo_loss = val_homo_loss_total / val_batches
        avg_val_lumo_loss = val_lumo_loss_total / val_batches
        avg_val_r2_loss = val_r2_loss_total / val_batches

        # Logging aggregated metrics to wandb after each epoch
        logging.info("[Epoch %d] Train Loss: %.3f, KL: %.2f, Word Acc: %.2f, Topo Acc: %.2f, Assm Acc: %.2f, Prop Loss: %.2f", 
                     epoch_num + 1, avg_train_loss, avg_train_meters[0], avg_train_meters[1], avg_train_meters[2], avg_train_meters[3], avg_train_prop_loss)
        logging.info("[Epoch %d] Val Loss: %.3f, KL: %.2f, Word Acc: %.2f, Topo Acc: %.2f, Assm Acc: %.2f, Prop Loss: %.2f",
                     epoch_num + 1, avg_val_loss, avg_val_meters[0], avg_val_meters[1], avg_val_meters[2], avg_val_meters[3], avg_val_prop_loss)

        if wandb_flag == True:
            wandb.log({
                "Epoch": epoch_num + 1,
                "Train Loss": avg_train_loss,
                "Train KL Divergence": avg_train_meters[0],
                "Train Word Accuracy": avg_train_meters[1],
                "Train Topo Accuracy": avg_train_meters[2],
                "Train Assembly Accuracy": avg_train_meters[3],
                "Train Property Loss": avg_train_prop_loss,
                "Train Homo Loss": avg_train_homo_loss,
                "Train Lumo Loss": avg_train_lumo_loss,
                "Train R2 Loss": avg_train_r2_loss,
                "Val Loss": avg_val_loss,
                "Val KL Divergence": avg_val_meters[0],
                "Val Word Accuracy": avg_val_meters[1],
                "Val Topo Accuracy": avg_val_meters[2],
                "Val Assembly Accuracy": avg_val_meters[3],
                "Val Property Loss": avg_val_prop_loss,
                "Val Homo Loss": avg_val_homo_loss,
                "Val Lumo Loss": avg_val_lumo_loss,
                "Val R2 Loss": avg_val_r2_loss,
                "Learning Rate": scheduler.get_last_lr()[0],
                "Beta": beta,
                "Best Epoch": best_epoch
            })

        # Save model checkpoint
        if (epoch_num + 1) % save_iter == 0:
            model_checkpoint_path = os.path.join(save_dir, f"model.epoch-{epoch_num + 1}.pth")
            torch.save(model.state_dict(), model_checkpoint_path)
            
            # Remove old checkpoints, keeping only the latest and best model
            remove_old_checkpoints(save_dir)

        # Early stopping
        if kl_active:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_epoch = epoch_num
                torch.save(model.state_dict(), save_dir + f"/model.best")
            else:
                patience_counter += 1
                print("No improvement in:", patience_counter)
                if patience_counter >= early_stopping_patience:
                    logging.info("Early stopping triggered. No improvement in validation loss for %d epochs.", early_stopping_patience)
                    break
        else:
            logging.info("KL divergence not yet active. Skipping early stopping check.")
            print("total_step: ", total_step)

    return model

if __name__ == '__main__':
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='../data/exlcuded1_train/')
    parser.add_argument('--test', default='../data/exlcuded1_test/')
    parser.add_argument('--vocab', default='../data/excluded1_vocab.txt')
    parser.add_argument('--save_dir', default='./excluded1')
    parser.add_argument('--csv_file', default='../data/excluded1_qm9_smiles_prop_normalized.csv') #should be normalized csv
    parser.add_argument('--wandb_flag', type=bool, default=True)
    parser.add_argument('--wandb_name', default='excluded1')
    parser.add_argument('--load_epoch', type=int, default=0)
    parser.add_argument('--z_prop_size', type=int, default=14) 

    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--latent_size', type=int, default=56)
    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_norm', type=float, default=50.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--step_beta', type=float, default=0.0001)
    parser.add_argument('--max_beta', type=float, default=0.01)
    parser.add_argument('--warmup', type=int, default=1500) #how many steps before kl activated (for 10k 1 epoch = 100 step)

    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--anneal_rate', type=float, default=0.9) #for lr schedule
    parser.add_argument('--anneal_iter', type=int, default=1000) #for lr schedule
    parser.add_argument('--kl_anneal_iter', type=int, default=700)  
    parser.add_argument('--print_iter', type=int, default=5)
    parser.add_argument('--save_iter', type=int, default=20)

    parser.add_argument('--property_weight', type=int, default=1)

    parser.add_argument('--early_stopping_patience', type=int, default=30)

    args = parser.parse_args()

    # Create save directory if it doesn't exist
    if os.path.isdir(args.save_dir) is False:
        os.makedirs(args.save_dir)

    # Save arguments as a configuration file
    config_file = os.path.join(args.save_dir, 'config.json')
    save_args_as_config(args, config_file)

    # Set up logging to file
    log_file = os.path.join(args.save_dir, 'train_log.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[FlushFileHandler(log_file), logging.StreamHandler()])

    print(args.wandb_flag)

    main_vae_train(args.train,
                   args.test,
                   args.vocab,
                   args.save_dir,
                   args.csv_file,
                   args.wandb_flag,
                   args.wandb_name,
                   args.load_epoch,
                   args.hidden_size,
                   args.batch_size,
                   args.latent_size,
                   args.depthT,
                   args.depthG,
                   args.lr,
                   args.clip_norm,
                   args.beta,
                   args.step_beta,
                   args.max_beta,
                   args.warmup,
                   args.epoch, 
                   args.anneal_rate,
                   args.anneal_iter, 
                   args.kl_anneal_iter,
                   args.print_iter, 
                   args.save_iter,
                   args.early_stopping_patience,
                   args.property_weight,
                   args.z_prop_size)
