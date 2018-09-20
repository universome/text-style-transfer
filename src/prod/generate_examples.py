#!/usr/bin/env python
import sys
import time

from tqdm import tqdm

from news import NEWS
from dialog_model import predict

output_file_path = sys.argv[1]
use_gold = bool(sys.argv[2]) if len(sys.argv) > 2 else False

N_LINES = 7
titles = [n['title'] for n in NEWS]
temperatures = [0.01, 0.1, 0.25, 0.5, 1., 2., 5., 10.]

with open(output_file_path, 'w') as f:
    for t in tqdm(temperatures):
        start_time = time.time()

        if not use_gold:
            dialogs = predict(titles, N_LINES, t)
        else:
            dialogs = []

            for news in NEWS:
                article = [s + '.' for p in news['text'].splitlines() for s in p.split('. ')]
                article[-1] = article[-1][:-1] # Removing last dot from double dot
                dialog = [article[0]]

                for i in range(1, len(article)):
                    condition = '|'.join(article[:i])
                    predicted = predict([condition], 1, t, 'sample')
                    next_line = predicted[0][-1]['text']
                    dialog.append(next_line)

                dialogs.append(dialog)

        elapsed = time.time() - start_time

        f.write('\n\n====== Temperature: {}. (Took seconds: {:.03f}) ======\n\n'.format(t, elapsed))

        for dialog in dialogs:
            for line in dialog:
                assert type(line) is str
                f.write(line + '\n')
            f.write('---------------------------------\n')

        f.flush()
