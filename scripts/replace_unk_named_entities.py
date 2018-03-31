#!/usr/bin/env python

import sys

from tqdm import tqdm; tqdm.monitor_interval = 0

NON_SEPARATING_PUNCTUATION = {',', ';', ':', '(', ')', '\'', '"', '_'}
UNK_NE_TOKEN = '__NE_UNK__'

def replace_unk_nes(s):
    if len(s) == 0: return s

    words = s.split()

    for i, word in enumerate(words[1:]):
        if not any(c.isupper() for c in word): continue
        if word.startswith('__NE_'): continue  # Our NER has successfully recognized this word

        # We get prev_c from word at index i, and not (i-1),
        # because our loop starts from "[1:]"
        prev_c = words[i][-1]

        if prev_c.isalnum() or (prev_c in NON_SEPARATING_PUNCTUATION):
            words[i+1] = UNK_NE_TOKEN

    s = ' '.join(words)

    return s

def main(input_file_path, output_file_path):
    print('Input file path:', input_file_path)
    print('Output file path:', output_file_path)

    with open(input_file_path, 'r', encoding='utf-8') as file_in:
        with open(output_file_path, 'w', encoding='utf-8') as file_out:
            for i, line in tqdm(enumerate(file_in)):
                file_out.write(replace_unk_nes(line.replace('\n', '')) + '\n')

                if i % 1000 == 0: file_out.flush()

if __name__ == '__main__':
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    main(input_file_path, output_file_path)

