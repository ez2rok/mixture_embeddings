#!/bin/bash
# Download the Greengenes dataset as a FASTA file
# More info: https://greengenes.secondgenome.com/?prefix=downloads/greengenes_database/gg_12_10/

# Specify the url and destination
url='https://gg-sg-web.s3-us-west-2.amazonaws.com/downloads/greengenes_database/gg_12_10/gg_12_10.fasta.gz'
destination='data/raw/greengenes'

# Make directory if it doesn't exist
mkdir -p $destination

# Download greengenes and unzip it in the `data/raw/greengenes` directory
wget -v $url -P $destination
gzip -dkrvc $destination/gg_12_10.fasta.gz > $destination/gg_12_10.fasta
rm data/raw/greengenes/gg_12_10.fasta.gz