#!/bin/bash

mosesdecoder="ext-libs/mosesdecoder"
multi30k_data_dir="data/multi30k"
europarl_data_dir="data/europarl-v7"
generated_data_dir="data/generated"

threads=6

mkdir -p $generated_data_dir

# Tokenizing multi30k
for file in $(ls "$multi30k_data_dir")
do
    lang="${file: -2}"
    cat "$multi30k_data_dir/$file" | \
        $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $lang | \
        $mosesdecoder/scripts/tokenizer/tokenizer.perl -threads $threads -l $lang > \
        $generated_data_dir/multi30k.$file.tok
done

