#!/bin/bash
CONDAENV=zulu
CLASSIFIERCMDFILE="../prod/classifier/cmd.txt"
DETECTORCMDFILE="../prod/detector/cmd.txt"
SEEDFILE="../prod/seeds/seeds.txt"

echo "Conda environment name: $CONDAENV"
echo "Classifier command filepath: $CLASSIFIERCMDFILE"
echo "Detector command filepath: $DETECTORCMDFILE"
echo "Seed filepath: $SEEDFILE"

CLASSIFIERCMD=$(head -n 1 $CLASSIFIERCMDFILE)
DETECTORCMD=$(head -n 1 $DETECTORCMDFILE)
SEED=$(head -n 1 $SEEDFILE)

echo "Classifier command: $CLASSIFIERCMD"
echo "Detector command: $DETECTORCMD"
echo "Seed: $SEED"

echo "Running classifier..."

# conda run -n $CONDAENV python -m predict $CLASSIFIERCMD --seed $seed

echo "Running object detector..."

# conda run -n $CONDAENV python -m predict $DETECTORCMD --seed $seed

echo "Done."