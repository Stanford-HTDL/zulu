#!/bin/bash
CONDAENV=zulu
CLASSIFIERCMDFILE="prod/classifier/cmd.txt"
DETECTORCMDFILE="prod/detector/cmd.txt"
SEEDFILE="prod/seeds/seeds.txt"

# read -p "Conda environment name: " CONDAENV
# read -p "Path to file containing command: " CMDFILE
# read -p "Path to file containing seeds: " SEEDFILE

echo "Conda environment name: $CONDAENV"
echo "Classifier command filepath: $CLASSIFIERCMDFILE"
echo "Detector command filepath: $DETECTORCMDFILE"
echo "Seed filepath: $SEEDFILE"

# read -p "Continue? (Y/N): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1

CLASSIFIERCMD=(head -n 1 $CLASSIFIERCMDFILE)
DETECTORCMD=(head -n 1 $DETECTORCMDFILE)
SEED=(head -n 1 $SEEDFILE)

echo "Classifier command: $CLASSIFIERCMD"
echo "Detector command: $DETECTORCMD"
echo "Seed: $SEED"

echo "Running classifier..."

# conda run -n $CONDAENV python -m predict $CLASSIFIERCMD --seed $seed

echo "Running object detector..."

# conda run -n $CONDAENV python -m predict $DETECTORCMD --seed $seed

echo "Done."