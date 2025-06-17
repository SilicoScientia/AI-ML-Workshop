import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re
import subprocess
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import lime
import lime.lime_tabular
from pathlib import Path

__all__ = ['convert_pdbqt_to_mol2', 'prepare_data', 'train_model']

# Define the GCN model
class GCNModel(torch.nn.Module):
    def __init__(self, num_features, hidden_channels=64):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

def convert_pdbqt_to_pdb(pdbqt_file):
    """Convert PDBQT file to PDB format using OpenBabel"""
    # Create temporary file for PDB output
    temp_pdb = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
    temp_pdb.close()
    
    # Convert PDBQT to PDB
    cmd = f'obabel {pdbqt_file} -O {temp_pdb.name} --gen3D'
    print(f"Converting PDBQT to PDB: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0 and os.path.exists(temp_pdb.name) and os.path.getsize(temp_pdb.name) > 0:
        print("Conversion successful")
        return temp_pdb.name
    
    print("Conversion failed")
    return None

def extract_binding_energy(pdbqt_file):
    """Extract binding energy from Vina PDBQT output file"""
    with open(pdbqt_file, 'r') as f:
        for line in f:
            if line.startswith('REMARK VINA RESULT:'):
                try:
                    # The binding energy is the first number after 'RESULT:'
                    parts = line.split()
                    energy_index = parts.index('RESULT:') + 1
                    energy = float(parts[energy_index])
                    print(f"Found binding energy: {energy}")
                    return energy
                except (IndexError, ValueError) as e:
                    print(f"Error parsing energy from line: {line.strip()}")
                    print(f"Error details: {str(e)}")
                    return None
    print("No binding energy found in file")
    return None

def process_molecule(mol):
    """Process molecule to handle aromaticity and other issues"""
    if mol is None:
        return None
        
    try:
        # First attempt: Try to sanitize without modifying aromaticity
        try:
            Chem.SanitizeMol(mol)
        except:
            print("Initial sanitization failed, trying alternative approaches...")
            
            # Second attempt: Try to kekulize
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
                Chem.SanitizeMol(mol)
            except:
                print("Kekulization failed, trying to remove aromaticity...")
                
                # Third attempt: Remove aromaticity flags
                for atom in mol.GetAtoms():
                    atom.SetIsAromatic(False)
                for bond in mol.GetBonds():
                    bond.SetIsAromatic(False)
                Chem.SanitizeMol(mol)
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 2D coordinates if not present
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)
        
        # Final sanitization
        Chem.SanitizeMol(mol)
        
        print(f"Successfully processed molecule with {mol.GetNumAtoms()} atoms")
        return mol
    except Exception as e:
        print(f"Error processing molecule: {str(e)}")
        return None

def mol_to_graph(mol):
    """Convert RDKit molecule to graph representation"""
    if mol is None:
        return None, None
    
    try:
        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            # Basic atom features
            features = [
                atom.GetAtomicNum(),  # Atomic number
                atom.GetDegree(),     # Number of bonds
                atom.GetFormalCharge(),  # Formal charge
                atom.GetHybridization().real,  # Hybridization
                atom.GetIsAromatic(),  # Aromaticity
                atom.GetTotalNumHs(),  # Number of hydrogens
                atom.GetNumRadicalElectrons(),  # Radical electrons
                atom.GetIsAromatic()  # Aromaticity (duplicated for emphasis)
            ]
            atom_features.append(features)
        
        # Get edge indices
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # Add both directions for undirected graph
            edge_index.append([i, j])
            edge_index.append([j, i])
        
        return torch.tensor(atom_features, dtype=torch.float), torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    except Exception as e:
        print(f"Error creating graph: {str(e)}")
        return None, None

def prepare_data(csv_path, best_poses_dir):
    """Prepare data for training"""
    print(f"Reading CSV file from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} entries in CSV file")
    print("\nCSV file columns:", df.columns.tolist())
    print("\nFirst few rows of CSV:")
    print(df.head())
    
    # Prepare data list
    data_list = []
    
    # List all PDBQT files
    pdbqt_files = [f for f in os.listdir(best_poses_dir) if f.endswith('_best.pdbqt')]
    print(f"\nFound {len(pdbqt_files)} PDBQT files in {best_poses_dir}")
    print("First few PDBQT files:", pdbqt_files[:5])
    
    # Process each PDBQT file in the best_poses directory
    for pdbqt_file in pdbqt_files:
        print(f"\nProcessing: {pdbqt_file}")
        pdbqt_path = os.path.join(best_poses_dir, pdbqt_file)
        
        # Extract binding energy
        binding_energy = extract_binding_energy(pdbqt_path)
        if binding_energy is None:
            print(f"Warning: Could not extract binding energy from {pdbqt_file}")
            continue
        
        # Convert PDBQT to PDB
        print("Converting PDBQT to PDB...")
        pdb_path = convert_pdbqt_to_pdb(pdbqt_path)
        if pdb_path is None:
            print(f"Warning: Could not convert {pdbqt_file} to PDB format")
            continue
        
        try:
            # Read PDB file with RDKit
            print("Reading PDB file with RDKit...")
            mol = Chem.MolFromPDBFile(pdb_path)
            
            if mol is None:
                print(f"Warning: Could not read PDB file for {pdbqt_file}")
                continue
            
            # Process the molecule
            mol = process_molecule(mol)
            if mol is None:
                print(f"Warning: Could not process molecule for {pdbqt_file}")
                continue
            
            print(f"Successfully created molecule with {mol.GetNumAtoms()} atoms")
            
            # Convert molecule to graph
            print("Converting molecule to graph...")
            x, edge_index = mol_to_graph(mol)
            if x is None or edge_index is None:
                print(f"Warning: Could not create graph for {pdbqt_file}")
                continue
                
            print(f"Successfully created graph with {x.size(0)} atoms and {edge_index.size(1)//2} bonds")
            
            # Create PyTorch Geometric data object
            data = Data(x=x, edge_index=edge_index, y=torch.tensor([binding_energy], dtype=torch.float))
            data_list.append(data)
            print(f"Successfully added data for {pdbqt_file}")
            
        except Exception as e:
            print(f"Error processing {pdbqt_file}: {str(e)}")
        finally:
            # Clean up temporary PDB file
            if os.path.exists(pdb_path):
                os.remove(pdb_path)
    
    print(f"\nTotal valid data points prepared: {len(data_list)}")
    if len(data_list) == 0:
        raise ValueError("No valid data points were prepared. Please check your input files and paths.")
    
    return data_list

def train_model(data_list, num_epochs=200):
    # Split data into train and validation sets
    train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)
    
    # Initialize model
    num_features = train_data[0].x.size(1)
    model = GCNModel(num_features=num_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_data:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long))
            loss = F.mse_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_data:
                out = model(data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long))
                val_loss += F.mse_loss(out, data.y).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_gcn_model.pt')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_data):.4f}, Val Loss: {val_loss/len(val_data):.4f}')

def evaluate_model(model, data_list, save_dir='model_output'):
    """
    Evaluate the model performance and save results
    
    Args:
        model: Trained GCN model
        data_list: List of PyTorch Geometric data objects
        save_dir: Directory to save evaluation results
    """
    # Create output directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists to store predictions and actual values
    all_predictions = []
    all_actual = []
    
    # Make predictions
    with torch.no_grad():
        for data in data_list:
            out = model(data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long))
            all_predictions.extend(out.cpu().numpy())
            all_actual.extend(data.y.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions).flatten()
    actual = np.array(all_actual).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)
    
    # Create metrics dictionary
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    # Save metrics to CSV
    pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, 'model_metrics.csv'), index=False)
    
    # Create and save plots
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predictions, alpha=0.5)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
    plt.xlabel('Actual Binding Energy')
    plt.ylabel('Predicted Binding Energy')
    plt.title('Actual vs Predicted Binding Energy')
    plt.savefig(os.path.join(save_dir, 'actual_vs_predicted.png'))
    plt.close()
    
    # Create residual plot
    residuals = predictions - actual
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Binding Energy')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig(os.path.join(save_dir, 'residual_plot.png'))
    plt.close()
    
    # Create distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Count')
    plt.title('Distribution of Residuals')
    plt.savefig(os.path.join(save_dir, 'residual_distribution.png'))
    plt.close()
    
    # Print results
    print("\nModel Evaluation Results:")
    print("------------------------")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"\nResults have been saved to the '{save_dir}' directory")
    
    return metrics

def main():
    try:
        print("Starting data preparation...")
        # Prepare data and train the model
        data_list = prepare_data('docking_outputs/docking_results.csv', 'docking_outputs/best_poses')
        
        print("\nStarting model training...")
        train_model(data_list)
        
        print("\nLoading best model for evaluation...")
        # Load the best model
        num_features = data_list[0].x.size(1)
        model = GCNModel(num_features=num_features)
        model.load_state_dict(torch.load('best_gcn_model.pt'))
        
        print("\nEvaluating model performance...")
        evaluate_model(model, data_list)
        
        print("\nTraining and evaluation completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nPlease check that:")
        print("1. The CSV file exists at 'docking_outputs/docking_results.csv'")
        print("2. The best poses directory exists at 'docking_outputs/best_poses'")
        print("3. All required packages are installed")
        print("4. The PDBQT files are in the correct format")

if __name__ == "__main__":
    main() 