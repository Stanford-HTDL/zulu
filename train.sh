#!/bin/bash
CONDAENV=zulu

read -p "Path to file containing command: " CMDFILE
read -p "Path to file containing seeds: " SEEDFILE

echo "Command filepath: $CMDFILE"
echo "Seed filepath: $SEEDFILE"

read -p "Continue? (Y/N): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1

while read line; do    
    echo "Seed: $line"    
done < $SEEDFILE
# conda run -n $CONDAENV python -m train