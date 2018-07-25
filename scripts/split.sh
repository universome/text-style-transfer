#!/bin/bash

mosesdecoder="ext-libs/mosesdecoder"
input_file=$1
output_file=$2
lang=${3:-ru}
threads=${4:-20}

echo "Splitting $input_file into $output_file"
echo "Language is $lang, threads is $threads"

$mosesdecoder/scripts/ems/support/split-sentences.perl \
    -l $lang -threads $threads < $input_file > $output_file
