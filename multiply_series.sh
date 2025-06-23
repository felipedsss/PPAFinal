#/bin/bash
mkdir -p bckp
#BACKUP CSV FILES
cp empresas/*.csv bckp/
#MULTIPLY CSV FILES
for file in empresas/*.csv; do
    for i in {2..10}; do
        # Create a new file name with the iteration number
        new_file="empresas/$(basename "$file" .csv)_$i.csv"
        cp "$file" "$new_file"  # Copy the original file to the new file
        # Multiply the contents of the file by the iteration number and save to the new file
        
    done
done