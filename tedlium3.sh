#!/bin/bash


# Script for converting TEDLIUM data from sph to flac.
# It takes a while, but saves tons of time in the long-run
BASE_DIR="/media/drive1/jaydenfassett/audio_data/TEDLIUM_release-3"
SUBSETS=("dev" "test" "train")

for subset in "${SUBSETS[@]}"; do
    # Input SPH folder for this subset
    IN_SPH_DIR="$BASE_DIR/legacy/$subset/sph"

    # Count total .sph files in this subset
    total_files=$(find "$IN_SPH_DIR/" -type f -name "*.sph" 2>/dev/null | wc -l)
    count=0

    # Convert each .sph to .flac in place
    for file in "$IN_SPH_DIR"/*.sph; do
        [[ -f "$file" ]] || continue  # Skip if no file
        ((count++))
        filename_no_ext="$(basename "$file" .sph)"
        out_file="$IN_SPH_DIR/$filename_no_ext.flac"
    
        # Skip if already converted
        if [[ -f "$out_file" ]]; then
            echo "[$count/$total_files] Skipping $filename_no_ext.sph (FLAC exists)."
        else
            echo -n "[$count/$total_files] Converting $filename_no_ext.sph... "
            ffmpeg -hide_banner -loglevel error \
                   -i "$file" -c:a flac "$out_file" -y \
            && echo "✔ Done" || echo "❌ Failed"
        fi
    done

    echo "$subset complete"

done

echo "Conversion complete."
