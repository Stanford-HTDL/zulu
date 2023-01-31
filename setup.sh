#!/bin/bash
### Miniconda setup ###
MINICONDA=https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
MINIOUTPATH=../miniconda.sh
CONDAENV=zulu
SETUPFILE=$CONDAENV.txt

wget -O $MINIOUTPATH $MINICONDA
bash $MINIOUTPATH
conda create --name $CONDAENV --file $SETUPFILE

### Git setup ###
# Uncomment the following lines if installing without pulling from git first
#sudo apt-get update
#sudo apt-get install git-all -y
#GITPATH=https://github.com/rcorrero/$CONDAENV.git
#git pull $GITPATH
#echo "__pycache__" >> .gitignore

### Download Eurosat dataset ###
EUROSATPATH=https://madm.dfki.de/files/sentinel/EuroSATallBands.zip
EUROSATOUTPATH=../eurosat.zip
wget -O $EUROSATOUTPATH $EUROSATPATH
sudo apt-get install unzip
unzip $EUROSATOUTPATH -d ../eurosat/

### Inpot necessary packages from PYPI ###
conda run -n $CONDAENV pip3 install light-pipe \
    && pip3 install Pillow