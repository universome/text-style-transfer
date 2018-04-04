#!/bin/bash

subword_nmt="ext-libs/subword-nmt"
data_dir="data/generated"
tmp_dir="$data_dir/_tmp"

data_src=$1
data_trg=$2
num_bpes=$3

echo "Source data: $data_src"
echo "Target data: $data_trg"
echo "Num bpes: $num_bpes"

mkdir -p $tmp_dir

bpes="$tmp_dir/bpes"
vocab_src="$tmp_dir/src.vocab"
vocab_trg="$tmp_dir/trg.vocab"

# Learning BPEs
python "$subword_nmt/learn_joint_bpe_and_vocab.py" --input $data_src $data_trg \
    -s $num_bpes -o $bpes --write-vocabulary $vocab_src $vocab_trg

# Applying BPEs
python "$subword_nmt/apply_bpe.py" -c $bpes < $data_src > $data_src.bpe
python "$subword_nmt/apply_bpe.py" -c $bpes < $data_trg > $data_trg.bpe

# Cleaning
rm -rf $tmp_dir

