#!/bin/bash

dataset=$1
num_bpes=$2

data_dir="data/generated"
tmp_dir="$data_dir/_tmp-$RANDOM"

echo "Dataset: $dataset"
echo "Num bpes: $num_bpes"

mkdir -p $tmp_dir

bpes="$tmp_dir/bpes"

subword-nmt learn-bpe -s $num_bpes < $dataset > $bpes
subword-nmt apply-bpe -c $bpes < $dataset > $dataset.bpe

# Cleaning
rm -rf $tmp_dir
