# To DO
# Phillip wrote notes on this here:
# https://docs.google.com/document/d/1pwL26og4Rbf3nAT53qA1mFlMbBn3bQ4pqk7TvPILjbE/edit#

# I initially copied the files from his github repo
# (https://github.com/pchlenski/mixture/tree/main/16s) and pasted them in the
# data/interim/moms_pi directory

# Run this script from mixture_embeddings directory
destination='data/raw/moms_pi'

mkdir -p $destination

unzip data/interim/moms_pi/16s.zip -d $destination
rm -rf data/interim/moms_pi/16s.zip