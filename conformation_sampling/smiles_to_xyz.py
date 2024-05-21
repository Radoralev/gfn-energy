import os
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem

# Function to generate conformers and save to XYZ file
def generate_conformer(smiles, output_path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error: Could not parse SMILES string: {smiles}")
        return
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    xyz_block = Chem.MolToXYZBlock(mol)
    with open(output_path, "w") as f:
        f.write(xyz_block)

# Function to process the input file
def process_file(input_file, base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    with open(input_file, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            fields = line.strip().split(';')
            if len(fields) < 2:
                print(f"Error: Unexpected line format: {line}")
                continue
            smiles = fields[1]
            safe_smiles = smiles.replace("/", "_").replace("\\", "_")
            folder_name = os.path.join(base_dir, safe_smiles)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            output_file = os.path.join(folder_name, "base.xyz")
            generate_conformer(smiles, output_file)
            print(f"Generated conformer for {smiles} at {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate conformers from a SMILES .txt file.")
    parser.add_argument("input_file", type=str, help="Path to the input .txt file.")
    parser.add_argument("output_dir", type=str, help="Directory where the 'conformers' folder should be placed.")
    args = parser.parse_args()
    
    input_file = args.input_file
    output_dir = args.output_dir
    base_dir = os.path.join(output_dir, "conformers")
    
    process_file(input_file, base_dir)

if __name__ == "__main__":
    main()
