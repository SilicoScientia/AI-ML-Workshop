import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import subprocess
import tempfile
import numpy as np

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

def predict_binding_energy(model, pdbqt_file):
    """Predict binding energy for a single PDBQT file"""
    print(f"\nProcessing file: {pdbqt_file}")
    
    # Convert PDBQT to PDB
    pdb_path = convert_pdbqt_to_pdb(pdbqt_file)
    if pdb_path is None:
        print(f"Failed to convert {pdbqt_file} to PDB format")
        return None
    
    try:
        # Read PDB file with RDKit
        mol = Chem.MolFromPDBFile(pdb_path)
        if mol is None:
            print(f"Failed to read PDB file for {pdbqt_file}")
            return None
        
        # Process the molecule
        mol = process_molecule(mol)
        if mol is None:
            print(f"Failed to process molecule for {pdbqt_file}")
            return None
        
        # Convert molecule to graph
        x, edge_index = mol_to_graph(mol)
        if x is None or edge_index is None:
            print(f"Failed to create graph for {pdbqt_file}")
            return None
        
        # Create PyTorch Geometric data object
        data = Data(x=x, edge_index=edge_index)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long))
            predicted_energy = out.item()
        
        return predicted_energy
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None
    finally:
        # Clean up temporary PDB file
        if os.path.exists(pdb_path):
            os.remove(pdb_path)

def main():
    # Load the trained model
    try:
        model = GCNModel(num_features=8)  # 8 features per atom
        model.load_state_dict(torch.load('best_gcn_model.pt'))
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Process all PDBQT files in the new_ligands folder
    new_ligands_dir = 'new_ligands'
    if not os.path.exists(new_ligands_dir):
        print(f"Error: {new_ligands_dir} directory not found")
        return
    
    # Get all PDBQT files
    pdbqt_files = [f for f in os.listdir(new_ligands_dir) if f.endswith('.pdbqt')]
    if not pdbqt_files:
        print(f"No PDBQT files found in {new_ligands_dir}")
        return
    
    print(f"\nFound {len(pdbqt_files)} PDBQT files in {new_ligands_dir}")
    
    # Store predictions
    predictions = []
    
    print("\nMaking predictions...")
    for pdbqt_file in pdbqt_files:
        print(f"\nProcessing: {pdbqt_file}")
        pdbqt_path = os.path.join(new_ligands_dir, pdbqt_file)
        
        predicted_energy = predict_binding_energy(model, pdbqt_path)
        if predicted_energy is not None:
            predictions.append({
                'ligand': pdbqt_file,
                'predicted_energy': predicted_energy
            })
            print(f"Predicted binding energy: {predicted_energy:.2f} kcal/mol")
        else:
            print(f"Failed to predict binding energy for {pdbqt_file}")
    
    # Save predictions to CSV
    if predictions:
        import pandas as pd
        df = pd.DataFrame(predictions)
        output_file = 'predicted_binding_energies.csv'
        df.to_csv(output_file, index=False)
        print(f"\nPredictions saved to {output_file}")
        
        # Display summary
        print("\nPrediction Summary:")
        print(f"Total ligands processed: {len(pdbqt_files)}")
        print(f"Successful predictions: {len(predictions)}")
        print(f"Failed predictions: {len(pdbqt_files) - len(predictions)}")
        
        # Display top 5 predictions
        print("\nTop 5 predicted binders (lowest energy):")
        print(df.sort_values('predicted_energy').head().to_string(index=False))
    else:
        print("\nNo successful predictions were made")

if __name__ == "__main__":
    main() 