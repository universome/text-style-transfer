#!/bin/bash

mosesdecoder="ext-libs/mosesdecoder"
input_file=$1
output_file=$2
threads=20

echo "Input file: $input_file"
echo "Output file: $output_file"
echo "Threads: $threads"

cat $input_file | \
    $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l ru | \
    $mosesdecoder/scripts/tokenizer/tokenizer.perl -threads $threads -l ru > \
    $output_file

