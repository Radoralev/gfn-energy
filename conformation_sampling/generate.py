import os
import subprocess
import json

# Directory containing molecule subdirectories
BASE_DIR = os.path.join(os.getcwd(), "conformers")

# JSON file
json_file = "conformers.json"
data = {}

# Save the initial directory to return to it later
initial_dir = os.getcwd()

# Iterate over each subdirectory in the base directory
for molecule_dir in os.listdir(BASE_DIR):
    molecule_path = os.path.join(BASE_DIR, molecule_dir)
    if os.path.isdir(molecule_path):
        # Find the .xyz file in the subdirectory
        xyz_file = os.path.join(molecule_path, "base.xyz")
        if os.path.isfile(xyz_file):
            print(f"Processing {xyz_file}...")
            
            # Create subfolders for solvation and vacuum
            solvation_dir = os.path.join(molecule_path, "solvation")
            vacuum_dir = os.path.join(molecule_path, "vacuum")
            os.makedirs(solvation_dir, exist_ok=True)
            os.makedirs(vacuum_dir, exist_ok=True)
            print(f"Created solvation directory: {solvation_dir}")
            print(f"Created vacuum directory: {vacuum_dir}")
            
            # Copy the .xyz file to the subfolders
            solvation_xyz = os.path.join(solvation_dir, "base.xyz")
            vacuum_xyz = os.path.join(vacuum_dir, "base.xyz")
            subprocess.run(["cp", xyz_file, solvation_xyz])
            subprocess.run(["cp", xyz_file, vacuum_xyz])
            print(f"Copied base.xyz to solvation directory: {solvation_xyz}")
            print(f"Copied base.xyz to vacuum directory: {vacuum_xyz}")
            
            # Run xtb and crest for solvation
            print(f"Navigating to solvation directory: {solvation_dir}")
            os.chdir(solvation_dir)
            subprocess.run(["xtb", "base.xyz", "--opt", "tight", "--gbsa", "h2o", "--gfn2"])
            subprocess.run(["crest", "xtbopt.xyz", "--gfn2", "--gbsa", "h2o", "-T", "8", "--noopt"])
            print(f"xtb and crest runs completed in solvation directory: {solvation_dir}")
            
            # Run xtb and crest for vacuum
            print(f"Navigating to vacuum directory: {vacuum_dir}")
            os.chdir(vacuum_dir)
            subprocess.run(["xtb", "base.xyz", "--opt", "tight", "--gfn2"])
            subprocess.run(["crest", "xtbopt.xyz", "--gfn2", "--gbsa", "h2o", "-T", "8", "--noopt"])
            print(f"xtb and crest runs completed in vacuum directory: {vacuum_dir}")
            
            # Return to the initial directory
            os.chdir(initial_dir)
            print(f"Returned to initial directory: {initial_dir}")
            
            # Add to JSON file
            data[molecule_dir] = {
                "solvated_conformers": os.path.join(solvation_dir, "output_solvated.log"),
                "nonsolvated_conformers": os.path.join(vacuum_dir, "output.log")
            }
            
            print(f"CREST run completed for {xyz_file}. Output saved in {molecule_path}/output.log")
        else:
            print(f"No .xyz file found in {molecule_path}. Skipping...")

# Write to JSON file
with open(json_file, 'w') as f:
    json.dump(data, f, indent=4)

print("Conformer generation completed for all molecules.")