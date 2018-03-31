#!/usr/bin/env python

import sys

import ner
from tqdm import tqdm; tqdm.monitor_interval = 0

extractor = ner.Extractor()

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

print('Input file path:', input_file_path)
print('Output file path:', output_file_path)


def replace_nes(s):
    try:
        for m in reversed(list(extractor(s))):
            s = s[:m.span.start] + '__NE_' + m.type + '__' + s[m.span.end:]
    except Exception as e:
        print('Could not extract NEs from a sentence:', s)
        print(e)

    return s


with open(input_file_path, 'r', encoding='utf-8') as file_in:
    with open(output_file_path, 'w', encoding='utf-8') as file_out:
        for i, line in tqdm(enumerate(file_in)):
            file_out.write(replace_nes(line))

            if i % 1000 == 0: file_out.flush()

