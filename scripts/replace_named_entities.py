#!/usr/bin/env python

import sys

import ner
from tqdm import tqdm; tqdm.monitor_interval = 0


def replace_nes(s, extractor):
    try:
        for m in reversed(list(extractor(s))):
            s = s[:m.span.start] + '__NE_' + m.type + '__' + s[m.span.end:]
    except Exception as e:
        print('Could not extract NEs from a sentence:', s)
        print(e)

    return s


def fix_quotes(s):
    # We normalize punctuation with mosesdecoder
    # and replaced quotes breaks NER. So let's fix this
    return s.replace('&quot;', '"')


def main(input_file_path, output_file_path):
    print('Input file path:', input_file_path)
    print('Output file path:', output_file_path)

    extractor = ner.Extractor()

    with open(input_file_path, 'r', encoding='utf-8') as file_in:
        with open(output_file_path, 'w', encoding='utf-8') as file_out:
            for i, line in tqdm(enumerate(file_in)):
                line = line.replace('\n', '') # just in case if our NER does not like line breaks
                line = fix_quotes(line)
                line = replace_nes(line, extractor)

                file_out.write(line + '\n')

                if i % 1000 == 0: file_out.flush()


if __name__ == '__main__':
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    main(input_file_path, output_file_path)

