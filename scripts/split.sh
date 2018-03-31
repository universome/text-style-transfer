#!/bin/bash

mosesdecoder="ext-libs/mosesdecoder"
threads=20
input_file=$1
output_file=$2

echo "Splitting $input_file into $output_file"

$mosesdecoder/scripts/ems/support/split-sentences.perl -l ru -threads $threads \
    < $input_file > $output_file

