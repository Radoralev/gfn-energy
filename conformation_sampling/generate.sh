#!/bin/bash

# Directory containing molecule subdirectories
BASE_DIR="/nfs/homedirs/ralev/gfn-energy/conformation_sampling/conformers"

# JSON file
json_file="conformers.json"
echo "{" > $json_file

# Iterate over each subdirectory in the base directory
for molecule_dir in "conformers"/*; do
    echo $molecule_dir
    if [ -d "$molecule_dir" ]; then
        # Find the .xyz file in the subdirectory
        xyz_file=$(find "$molecule_dir" -name "base.xyz" | head -n 1)
        #echo $xyz_file
        if [ -f "$xyz_file" ]; then
            echo "Processing $xyz_file..."
            
            # Create subfolders for solvation and vacuum
            mkdir -p "$molecule_dir/solvation"
            mkdir -p "$molecule_dir/vacuum"
            
            # Copy the .xyz file to the subfolders
            cp "$xyz_file" "$molecule_dir/solvation"
            cp "$xyz_file" "$molecule_dir/vacuum"
            
            # Run xtb on the .xyz file twice, once with --gbsa h2o and once without
            cd "$molecule_dir/solvation"
            xtb "base.xyz" --opt tight --gbsa h2o --gfn2
            crest xtbopt.xyz --gfn2 --gbsa h2o -T 8 --noopt

            cd "$molecule_dir/vacuum"
            xtb "base.xyz" --opt tight --gfn2
            crest xtbopt.xyz --gfn2 --gbsa h2o -T 8 --noopt
            
            # Run CREST on the optimized .xyz files
            
            # Add to JSON file
            echo "\"$(basename $molecule_dir)\": {\"solvated_conformers\": \"$molecule_dir/solvation/output_solvated.log\", \"nonsolvated_conformers\":\"$molecule_dir/vacuum/output.log\"}," >> $json_file
            
            echo "CREST run completed for $xyz_file. Output saved in $molecule_dir/output.log"
        else
            echo "No .xyz file found in $molecule_dir. Skipping..."
        fi
    fi
done

# Close JSON file
echo "}" >> $json_file

echo "Conformer generation completed for all molecules."