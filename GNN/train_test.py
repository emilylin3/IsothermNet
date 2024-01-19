import os
import psutil 
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv, GATConv, global_add_pool

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import optuna

from config import *
from cgcnn_att import CGCNNModel
    
    
"""
Training, testing, and validation of CGCNN model
"""

# Get system memory information
mem = psutil.virtual_memory()
print(f"Total Memory: {mem.total / (1024 ** 3):.2f} GB", flush=True)
print(f"Available Memory: {mem.available / (1024 ** 3):.2f} GB", flush=True)

class TrainCGCNN():
    """
    Train crystal graph convolutional neural network (CGCNN)
    """

    def __init__(self):      
        # Initialize general parameters
        if not os.path.exists(jobPath):
            os.makedirs(jobPath)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        print(self.device, flush=True)
        
        if train:
            if run_dataProcess:
                _, structureInputs = dataProcess_isotherm.structure_inputs(dataDir=jobPath)
            else:
                print("Loading structural data...", flush=True)
                structureInputs = torch.load(jobPath+"X_dataset_electro_xyz_bond_struc.pth")
                
            print("Building train, validation, and test data...", flush=True)
            nodeFeat, IMFeat, strucGlobalFeat, bondFeat, pressureFeat = structureInputs["nodeFeat"][:,:,:-19], structureInputs["IMFeat"], structureInputs["strucGlobalFeat"], structureInputs["bondFeat"], structureInputs["nodeFeat"][:,:,-19:]
            bondFeat = bondFeat[:, :nodeFeat.size(1), :nodeFeat.size(1)]
            
            texturalProp_data = pd.read_excel(jobPath + "texturalProperties_vol.xlsx")
            texturalFeat = torch.tensor(texturalProp_data.iloc[:,1:].values, dtype=torch.float32)
            texturalFeat[:, 0] = texturalFeat[:, 0]/max(texturalFeat[:, 0])
            texturalFeat[:, 5] = texturalFeat[:, 5]/max(texturalFeat[:, 5])
            
            pressureFeat = pressureFeat[:, 0, :]            
            pressureFeat = torch.index_select(pressureFeat, dim=1, index=pos)

            H_data = torch.load("H_dataset.pth")["H"] * -R
            H_data = torch.index_select(H_data, dim=1, index=pos)
            
            loadData = torch.load(jobPath + "y_dataset19.pth")
            y_data = loadData["isotherm"]
            y_data = torch.index_select(y_data, dim=1, index=pos)
            
            if len(num) == 1:
                self.train_batchSize = 1
                self.val_batchSize = train_batchSize
                self.test_batchSize = train_batchSize

                nodeFeat = nodeFeat[num[0], :, :].unsqueeze(0)
                bondFeat = bondFeat[num[0], :, :].unsqueeze(0)
                pressureFeat = pressureFeat[num[0], :, :].unsqueeze(0)
                texturalFeat = texturalFeat[num[0], :].unsqueeze(0)
                H_data = H_data[num[0], :].unsqueeze(0)
                y_data = y_data[num[0], :].unsqueeze(0)

                y_train_isotherm = y_data.double()
                y_val_isotherm = y_data.double()  
                y_test_isotherm = y_data.double()  

                train_dataset = TensorDataset(nodeFeat, bondFeat, texturalFeat, pressureFeat, y_train_isotherm)
                val_dataset = TensorDataset(nodeFeat, bondFeat, texturalFeat, pressureFeat, y_val_isotherm)
                self.train_DataLoader = DataLoader(train_dataset, batch_size=self.train_batchSize, shuffle=True, pin_memory=True)
                self.val_DataLoader = DataLoader(val_dataset, batch_size=self.val_batchSize, shuffle=True, pin_memory=True)
            
            else:
                nodeFeat = nodeFeat[num[0]:num[1], :, :]
                IMFeat = IMFeat[num[0]:num[1], :, :]
                strucGlobalFeat = strucGlobalFeat[num[0]:num[1], :, :]
                
                bondFeat = bondFeat[num[0]:num[1], :, :]
                
                y_data = y_data[num[0]:num[1], :]
                
                texturalFeat = texturalFeat[num[0]:num[1], :]
                pressureFeat = pressureFeat[num[0]:num[1], :]
                
                H_data = H_data[num[0]:num[1], :]

                x_node_train, x_node_val, x_IM_train, x_IM_val, x_strucGlobal_train, x_strucGlobal_val, x_bond_train, x_bond_val, x_textural_train, x_textural_val, x_pressure_train, x_pressure_val, y_train, y_val = train_test_split(nodeFeat, IMFeat, strucGlobalFeat, bondFeat, texturalFeat, pressureFeat, y_data, test_size=0.2, random_state=42)
                x_node_train, x_node_test, x_IM_train, x_IM_test, x_strucGlobal_train, x_strucGlobal_test, x_bond_train, x_bond_test, x_textural_train, x_textural_test, x_pressure_train, x_pressure_test, y_train, y_test = train_test_split(x_node_train, x_IM_train, x_strucGlobal_train, x_bond_train, x_textural_train, x_pressure_train, y_train, test_size=0.2, random_state=42)
            
                train_dataset = TensorDataset(x_node_train, x_IM_train, x_strucGlobal_train, x_bond_train, x_textural_train, x_pressure_train, y_train)
                val_dataset = TensorDataset(x_node_val, x_IM_val, x_strucGlobal_val, x_bond_val, x_textural_val, x_pressure_val, y_val)
                self.train_DataLoader = DataLoader(train_dataset, batch_size=train_batchSize, shuffle=True, pin_memory=True)
                self.val_DataLoader = DataLoader(val_dataset, batch_size=val_batchSize, shuffle=True, pin_memory=True)
                
            self.nodeFeat = nodeFeat
            self.IMFeat = IMFeat
            self.strucGlobalFeat = strucGlobalFeat
            self.bondFeat = bondFeat
            self.texturalFeat = texturalFeat
            self.pressureFeat = pressureFeat
            self.y_data = y_data
            self.H_data = H_data

        if test:
            if len(num) == 1:
                test_dataset = TensorDataset(nodeFeat, IMFeat, strucGlobalFeat, bondFeat, texturalFeat, pressureFeat, y_test_isotherm)
                self.test_DataLoader = DataLoader(test_dataset, batch_size=self.test_batchSize, shuffle=False, pin_memory=True)
            else:
                test_dataset = TensorDataset(x_node_test, x_IM_test, x_strucGlobal_test, x_bond_test, x_textural_test, x_pressure_test, y_test)
                self.test_DataLoader = DataLoader(test_dataset, batch_size=test_batchSize, shuffle=False, pin_memory=True)
                
            self.x_node_test = x_node_test
            self.x_IM_test = x_IM_test
            self.x_strucGlobal_test = x_strucGlobal_test
            self.x_textural_test = x_textural_test
            self.y_test = y_test
            
        self.N = nodeFeat.size(1)            # max number of nodes across all crystal structures
            
        print("Done building train, validation, and test data.", flush=True)
    
    def tune_hp(self):
        def objective_func(hp_params):
            N = self.N                     
                
            lr = hp_params.suggest_float("lr", 1e-5, 1e-1, log=True)
            weight_decay = hp_params.suggest_float("weight_decay", 1e-6, 1, log=True)
            dropout = hp_params.suggest_float("dropout", 0.0, 0.5)
            n_heads = hp_params.suggest_int("n_heads", 1, 16)

            structureParams = {
                "dim_IMFeat": self.IMFeat.size(2)*2,                              
                "dim_strucGlobalFeat": self.strucGlobalFeat.size(2)*2, 
                "dim_texturalFeat": self.texturalFeat.size(1)*2, 
                "dim_pressureFeat": self.pressureFeat.size(1)*2, 

                "dim_in": self.nodeFeat.size(2),    # number of features 

                "n_convLayer": 2,
                "dim_out": [64, 32], 

                "n_attLayer": 2,
                "dim_att": [128, 64],  

                "n_hidLayer_pool": 3,
                "dim_hidFeat": [16, 8, 4],

                "dim_fc_out": self.y_data.size(1),

                "n_heads": n_heads,
                "dropout": dropout
            }

            model = CGCNNModel(structureParams).to(self.device)

            if optim in ["sgd", "SGD"]:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            elif optim in ["Adam", "adam"]:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif optim in ['Adamax', 'adamax']:
                optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Train
            train_mse = []
            train_mae = []            
            model.train()            
            for batch, (x_node_train, x_IM_train, x_strucGlobal_train, x_bond_train, x_textural_train, x_pressure_train, y_data_isotherm) in (enumerate(self.train_DataLoader)):                
                batch_size_train = x_node_train.size(0)  # batch size (in training loop - number of crystal structures in the batch)
                x_data_train = [x_node_train, x_IM_train, x_strucGlobal_train, x_bond_train, x_textural_train, x_pressure_train]

                batchAssign_train = torch.tensor([b for b in range(batch_size_train) for n in range(N)])

                y_pred_isotherm = model(x_data_train, batchAssign_train, n_heads).squeeze()
                y_pred_isotherm = y_pred_isotherm.unsqueeze(-1)                    

                mse, mae, huber = self.calcLoss(y_pred_isotherm, y_data_isotherm)

                optimizer.zero_grad()
                mse.backward()
                optimizer.step()

                train_mse.append(mse.item())
                train_mae.append(mae.item())

            train_mse_mean = np.mean(train_mse)
            train_mae_mean = np.mean(train_mae)

            # Validation
            val_mse = []
            val_mae = []
            model.eval()
            for batch, (x_node_val, x_IM_val, x_strucGlobal_val, x_bond_val, x_textural_val, x_pressure_val, y_data_isotherm) in enumerate(self.val_DataLoader):
                batch_size_val = x_node_val.size(0) 
                x_data_val = [x_node_val, x_IM_val, x_strucGlobal_val, x_bond_val, x_textural_val, x_pressure_val]

                batchAssign_val = torch.tensor([b for b in range(batch_size_val) for n in range(N)])
                y_pred_isotherm = model(x_data_val, batchAssign_val, n_heads).squeeze()
                y_pred_isotherm = y_pred_isotherm.unsqueeze(-1)   

                mse, mae, huber = self.calcLoss(y_pred_isotherm, y_data_isotherm)

                val_mse.append(mse.item())
                val_mae.append(mae.item())

            val_mse_mean = np.mean(val_mse)
            val_mae_mean = np.mean(val_mae)

            return val_mse_mean

        # Run optimization to tune hyperparameters at the beginning (TPESampler, CmaEsSampler, NSGAIISampler, QMCSampler)
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), study_name="Hyperparameter Tuning", direction="minimize")
        study.enqueue_trial({"lr": 0.0030507918415409343,  
                            "weight_decay": 3.4018446324990345e-5,
                            "dropout": 0.30318566438711475,
                            "n_heads": 13
            })
        study.optimize(lambda x: objective_func(x), n_trials=max_evals)
        hp_opt_dict = study.best_params
        loss_opt = study.best_value
        
        hp_opt = np.array(list(hp_opt_dict.values()))
        
        lr = hp_opt[0]
        weight_decay = hp_opt [1]
        dropout = hp_opt[2]
        n_heads = int(hp_opt[3])
        
        return lr, weight_decay, dropout, n_heads
    
    
    def calcLoss(self, y_pred_isotherm, y_target_isotherm):
        y_pred_isotherm = y_pred_isotherm.double()
        y_target_isotherm = y_target_isotherm.double()
        
        mse = nn.MSELoss()(y_pred_isotherm, y_target_isotherm)     
        mae = nn.L1Loss()(y_pred_isotherm, y_target_isotherm)   
        huber = nn.SmoothL1Loss()(y_pred_isotherm, y_target_isotherm)   

        return mse, mae, huber
    
    
    def train(self, lr, weight_decay, dropout, n_heads):        
        N = self.N                      # max number of nodes across all crystal structures
        best_mse = 1e10
        best_mae = 1e10
        
        train_mse_mean = []
        val_mse_mean = []
        train_mae_mean = []
        val_mae_mean = []
        
        iter_stop = 0

        structureParams = {
            "dim_IMFeat": self.IMFeat.size(2)*2,                                
            "dim_strucGlobalFeat": self.strucGlobalFeat.size(2)*2, 
            "dim_texturalFeat": self.texturalFeat.size(1)*2, 
            "dim_pressureFeat": self.pressureFeat.size(1)*2, 

            "dim_in": self.nodeFeat.size(2),    # number of features you input (node)

            "n_convLayer": 2,
            "dim_out": [64, 32],

            "n_attLayer": 2,
            "dim_att": [128, 64], 

            "n_hidLayer_pool": 3,
            "dim_hidFeat": [16, 8, 4],

            "dim_fc_out": self.y_data.size(1),

            "n_heads": n_heads,
            "dropout": dropout
        }

        model = CGCNNModel(structureParams).to(self.device)

        if optim in ["sgd", "SGD"]:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif optim in ["Adam", "adam"]:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optim in ['Adamax', 'adamax']:
            optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)

        if load_checkpoint:
            checkpoint = torch.load(os.path.join(jobPath, "checkpoint"+indx_str+".pth"))
            start_epoch = checkpoint["epoch"]-1
            train_mse_mean = checkpoint["train_mse_mean"]
            val_mse_mean = checkpoint["train_mse_mean"]
            train_mae_mean = checkpoint["train_mae_mean"]
            val_mae_mean = checkpoint["val_mae_mean"]
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            start_epoch = 0

        epoch_vec = []
        for epoch in tqdm(range(start_epoch, num_epoch)):
            if iter_stop < train_patience:                # patience
                iter_stop += 1
                epoch_vec.append(epoch)

                # Train
                train_mse = []
                train_mae = []            
                model.train()            
                for batch, (x_node_train, x_IM_train, x_strucGlobal_train, x_bond_train, x_textural_train, x_pressure_train, y_data_isotherm) in (enumerate(self.train_DataLoader)):                
                    batch_size_train = x_node_train.size(0)  # batch size (in training loop - number of crystal structures in the batch)
                    x_data_train = [x_node_train, x_IM_train, x_strucGlobal_train, x_bond_train, x_textural_train, x_pressure_train]

                    batchAssign_train = torch.tensor([b for b in range(batch_size_train) for n in range(N)])

                    y_pred_isotherm = model(x_data_train, batchAssign_train, n_heads).squeeze()
                    y_pred_isotherm = y_pred_isotherm.unsqueeze(-1)                    
                    
                    mse, mae, huber = self.calcLoss(y_pred_isotherm, y_data_isotherm)
                    
                    optimizer.zero_grad()
                    mse.backward()
                    optimizer.step()

                    train_mse.append(mse.item())
                    train_mae.append(mae.item())

                train_mse_mean.append(np.mean(train_mse))
                train_mae_mean.append(np.mean(train_mae))
                
                # Validation
                val_mse = []
                val_mae = []
                model.eval()
                for batch, (x_node_val, x_IM_val, x_strucGlobal_val, x_bond_val, x_textural_val, x_pressure_val, y_data_isotherm) in enumerate(self.val_DataLoader):
                    batch_size_val = x_node_val.size(0)  # batch size (in training loop - number of crystal structures in the batch)
                    x_data_val = [x_node_val, x_IM_val, x_strucGlobal_val, x_bond_val, x_textural_val, x_pressure_val]

                    batchAssign_val = torch.tensor([b for b in range(batch_size_val) for n in range(N)])
                    y_pred_isotherm = model(x_data_val, batchAssign_val, n_heads).squeeze()
                    y_pred_isotherm = y_pred_isotherm.unsqueeze(-1)   
                    
                    mse, mae, huber = self.calcLoss(y_pred_isotherm, y_data_isotherm)

                    val_mse.append(mse.item())
                    val_mae.append(mae.item())

                val_mse_mean.append(np.mean(val_mse))
                val_mae_mean.append(np.mean(val_mae))

                torch.save({
                        "epoch": epoch + 1,
                        "train_mse_mean": train_mse_mean,
                        "train_mae_mean": train_mae_mean,
                        "val_mse_mean": val_mse_mean,
                        "val_mae_mean": val_mae_mean,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "dropout": dropout,
                        "n_heads": n_heads
                    }, os.path.join(jobPath, "checkpoint"+indx_str+".pth"))

                if (np.mean(val_mae) < best_mae) and (np.mean(val_mse) < best_mse):  
                    best_mae = np.mean(val_mae)
                    best_mse = np.mean(val_mse)

                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "dropout": dropout,
                        "n_heads": n_heads
                    }, os.path.join(jobPath, "best_model"+indx_str+".pth"))
                    
                    iter_stop = 0
                    
            else:
                print("Model exceeded 50 iterations without improvement. BREAK.")
                break
                
        return model
                
    def load_best_model(self, model):
        best_model_path = os.path.join(jobPath, "best_model"+indx_str+".pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint["epoch"]
            lr = checkpoint["lr"]
            weight_decay = checkpoint["weight_decay"]
            dropout = checkpoint["dropout"]
            n_heads = checkpoint["n_heads"]
            print(f"Loaded best model from epoch {epoch}", flush=True)
        else:
            print("Best model checkpoint not found.", flush=True)
        
        return n_heads

    def test(self, model, n_heads):
        N = self.N  

        # Testing / Validation
        val_mse = []
        val_mae = []
        model.eval()
        for batch, (x_node_val, x_IM_val, x_strucGlobal_val, x_bond_val, x_textural_val, x_pressure_val, y_data_isotherm) in enumerate(self.test_DataLoader):
            batch_size_val = x_node_val.size(0)  # batch size (in training loop - number of crystal structures in the batch)
            x_data_val = [x_node_val, x_IM_val, x_strucGlobal_val, x_bond_val, x_textural_val, x_pressure_val]

            batchAssign_val = torch.tensor([b for b in range(batch_size_val) for n in range(N)])        # needs to be repeated batch_size times (0,0,0, ..., batch_size-1, batch_size-1, batch_size-1)

            y_pred_isotherm = model(x_data_val, batchAssign_val, n_heads).squeeze()
            y_pred_isotherm = y_pred_isotherm.unsqueeze(-1)   
            
            mse, mae, huber = self.calcLoss(y_pred_isotherm, y_data_isotherm)
            val_mse.append(mse.item())
            val_mae.append(mae.item())
            
            print(f"MSE: {float(mse):.4f}, MAE: {float(mae):.4f}")
                        
            P = np.array([1e3,5e3,1e4,5e4,1e5,2e5,3e5,4e5,5e5,7e5,1e6,1.5e6,2e6,2.5e6,3e6,3.5e6,4e6,4.5e6,5e6])*0.00001
            
            err_tot = (np.array(y_data_isotherm)) - (y_pred_isotherm.detach().numpy())
            err_LB_tot = (np.array(y_data_isotherm)) - (y_pred_isotherm.detach().numpy())
            err_UB_tot = (np.array(y_data_isotherm)) - (y_pred_isotherm.detach().numpy())
            
            for i in range(y_data_isotherm.size(0)):
                for j, err_j in enumerate(err_tot[i,:]):
                    if err_j > 0:
                        err_UB_tot[i,j] = 0
                    else:
                        err_LB_tot[i,j] = 0
                                                                        
            if batch == 0:
                y_pred_isotherm_tot = torch.empty(1,1)
                y_data_isotherm_tot = torch.empty(1,1)
                err_LB_arr = torch.empty(1,1)
                err_UB_arr = torch.empty(1,1)
                                
                y_pred_isotherm_tot = torch.cat((y_pred_isotherm_tot, y_pred_isotherm), dim=0)[1:]
                y_data_isotherm_tot = torch.cat((y_data_isotherm_tot, y_data_isotherm), dim=0)[1:]
                err_LB_arr =  torch.cat((err_LB_arr, err_LB_arr), dim=0)[1:]
                err_UB_arr =  torch.cat((err_UB_arr, err_UB_arr), dim=0)[1:]
                                
            else:
                y_pred_isotherm_tot = torch.cat((y_pred_isotherm_tot, y_pred_isotherm), dim=0)
                y_data_isotherm_tot = torch.cat((y_data_isotherm_tot, y_data_isotherm), dim=0)
                err_LB_arr =  torch.cat((err_LB_arr, err_LB_arr), dim=0)
                err_UB_arr =  torch.cat((err_UB_arr, err_UB_arr), dim=0)

        print(f"Average MSE: {np.mean(val_mse):.4f}, Average MAE: {np.mean(val_mae):.4f}", flush=True)
                
        torch.save({
            "y_pred_isotherm": y_pred_isotherm_tot,
            "y_data_isotherm": y_data_isotherm_tot,
            "err_LB": err_LB_arr,
            "err_UB": err_UB_arr
        }, os.path.join(jobPath, "results"+indx_str+".pth"))

        # Plot losses        
        plt.figure(figsize=(15,6))
        data_loss = torch.load(os.path.join(jobPath, "checkpoint"+indx_str+".pth"))
        epoch = data_loss["epoch"]
        train_mse_mean = data_loss["train_mse_mean"]
        val_mse_mean = data_loss["train_mse_mean"]
        train_mae_mean = data_loss["train_mae_mean"]
        val_mae_mean = data_loss["val_mae_mean"]
        
        epoch_vec = range(0, len(train_mae_mean))
                
        plt.subplot(1, 2, 1)
        plt.plot(epoch_vec, np.log(np.array(train_mse_mean)), alpha=0.4, color="blue", label="Training - MSE")
        plt.plot(epoch_vec, np.log(np.array(val_mse_mean)), alpha=0.4, color="red", label="Validation - MSE")
        plt.plot(epoch_vec, np.log(np.array(train_mae_mean)), color="blue", label="Training - MAE")
        plt.plot(epoch_vec, np.log(np.array(val_mae_mean)), color="red", label="Validation - MAE")
        plt.title("Loss", **font_title)
        plt.xlabel("Epoch", **font)
        plt.ylabel("Log Loss", **font)
        plt.legend(prop=font)
        
        # Plot parity plots: target vs prediction
        min_val = min(min(y_pred_isotherm_tot[:, 0].detach().numpy()), min(y_data_isotherm_tot[:, 0].detach().numpy()))
        max_val = max(max(y_pred_isotherm_tot[:, 0].detach().numpy()), max(y_data_isotherm_tot[:, 0].detach().numpy()))
        
        xy = np.vstack([y_pred_isotherm_tot[:, 0].detach().numpy(), y_data_isotherm_tot[:, 0].detach().numpy()])
        z = gaussian_kde(xy)(xy)
        
        colors = [np.array([119,142,176])/255, np.array([170,126,170])/255, np.array([104,30,120])/255]
        custom_cmap = LinearSegmentedColormap.from_list("cmap", colors, N=256)

        plt.subplot(1, 2, 2)
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        plt.plot(np.linspace(min_val, max_val), np.linspace(min_val, max_val), linestyle="dashed", color="black")
        plt.scatter(y_pred_isotherm_tot[:, 0].detach().numpy(), y_data_isotherm_tot[:, 0].detach().numpy(), c=z, s=10, cmap=custom_cmap)
        plt.colorbar().set_ticks([])
        plt.title(f"{P[pos]:.2f} bar(s)", **font_title)
        plt.xlabel("Predicted Uptake [g/kg]", **font)
        plt.ylabel("Target Uptake [g/kg]", **font)
        
        plt.savefig(f"loss_parity{indx_str}", dpi=300)
        
        # Parity + textural categorization
        min_val = min(min(y_pred_isotherm_tot[:, 0].detach().numpy()), min(y_data_isotherm_tot[:, 0].detach().numpy()))
        max_val = max(max(y_pred_isotherm_tot[:, 0].detach().numpy()), max(y_data_isotherm_tot[:, 0].detach().numpy()))
                
        labels = ["Normalized Surface Area", "Void Fraction", "PLD [$\AA$]", "LCD [$\AA$]", "Density [g/cm$^3$]", "Normalized Channel Volume"]
        
        plt.figure(figsize=(15,6))
        for i in range(len(labels)):
            plt.subplot(2, 3, i+1)
            plt.plot(np.linspace(min_val, max_val), np.linspace(min_val, max_val), linestyle="dashed", color="black")
            plt.scatter(y_pred_isotherm_tot[:, 0].detach().numpy(), y_data_isotherm_tot[:, 0].detach().numpy(), c=np.log(self.x_textural_test[:, i]), s=10, cmap="Spectral")
            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False) 
            for j in plt.colorbar().ax.yaxis.get_ticklabels():
                j.set_family("Arial")
            plt.title(f"{labels[i]}: {P[pos]:.2f} bar(s)", **font_title)
            if i in range(4, 8):
                plt.xlabel("Predicted Uptake [g/kg]", **font) 
            if i in [0, 4]:
                plt.ylabel("Target Uptake [g/kg]", **font)
        plt.savefig(f"{jobPath}parity_textural{indx_str}", dpi=300)
        
                  
    def PCA(self):
        x = torch.cat((self.strucGlobalFeat[:, 0, :], self.texturalFeat, self.pressureFeat), dim=-1)
        pca = PCA(n_components=2)
        pca_fit = pca.fit_transform(x)
        
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

        symmetry = self.strucGlobalFeat[:, 0, 3:10]
        for i in range(symmetry.size(1)):
            symmetry[:, i] = symmetry[:, i] * (i+1)
        symmetry = torch.sum(symmetry, dim=1).unsqueeze(-1)
        
        labels = ["Normalized Surface Area", "Void Fraction", "PLD [$\AA$]", "LCD [$\AA$]", "Density [g/cm$^3$]", 
                  "Normalized Channel Volume", "CO2 Uptake [g/kg]", "Heat of Adsorption [kJ/mol]"]
        data = torch.cat((self.texturalFeat[:, :-1], self.y_data, self.H_data), dim=-1)
        
        plt.figure(figsize=(15,6))
        for i in range(len(labels)):
            plt.subplot(2, 4, i+1)
            plt.scatter(pca_fit[:, 0], pca_fit[:, 1], c=data[:, i], s=10, cmap="Spectral")
            plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False) 
            for j in plt.colorbar().ax.yaxis.get_ticklabels():
                j.set_family("Arial")
            plt.title(f"{labels[i]}", **font_title)
            if i in range(4, 8):
                plt.xlabel("PC 1", **font) 
            if i in [0, 4]:
                plt.ylabel("PC 2", **font)
        plt.savefig(f"{jobPath}pca{indx_str}", dpi=300)
        
            
if __name__ == "__main__":    
    trainer = TrainCGCNN()
    trainer.PCA()
    
    if load_checkpoint or load_hp:
        # checkpoint = torch.load(os.path.join(jobPath, "checkpoint"+indx_str+".pth"))
        # lr, weight_decay, dropout, n_heads = checkpoint["lr"], checkpoint["weight_decay"], checkpoint["dropout"], checkpoint["n_heads"]
        
        lr = 0.0030507918415409343
        weight_decay = 3.4018446324990345e-5
        n_heads = 13
        dropout = 0.30318566438711475
        loss_opt = 0                         # dummy value
    else:
        lr, weight_decay, dropout, n_heads = trainer.tune_hp()
        
    model = trainer.train(lr, weight_decay, dropout, n_heads)
    n_heads = trainer.load_best_model(model)
    trainer.test(model, n_heads)
    