#!/bin/bash

# Directory containing the CSV files
input_dir="fed_results"

rm -f "$input_dir"/*.png

# Loop through all CSV files in the input directory
for csv_file in "$input_dir"/*.csv; do
    # Extract the base name of the file (without path and extension)
    base_name=$(basename "$csv_file" .csv)
    
    # Construct the output PNG file name
    png_file="${input_dir}/${base_name}.png"
    
    # Call the Python script
    python analyze_results.py --input "$csv_file" --output "$png_file" --rows 90
    
    echo "Processed $csv_file -> $png_file"
done

echo "All files processed."