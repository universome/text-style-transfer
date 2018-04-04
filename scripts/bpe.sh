#!/bin/bash

dataset=$1
num_bpes=$2

subword_nmt="ext-libs/subword-nmt"
data_dir="data/generated"
tmp_dir="$data_dir/_tmp-$RANDOM"

echo "Dataset: $dataset"
echo "Num bpes: $num_bpes"

mkdir -p $tmp_dir

bpes="$tmp_dir/bpes"

python "$subword_nmt/learn_bpe.py" -s $num_bpes < $dataset > $bpes
python "$subword_nmt/apply_bpe.py" -c $bpes < $dataset > $dataset.bpe

# Cleaning
rm -rf $tmp_dir

