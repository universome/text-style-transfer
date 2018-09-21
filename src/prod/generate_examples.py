#!/usr/bin/env python
import sys
import time

from tqdm import tqdm

from news import NEWS
from dialog_model import predict as generate_dialog
from style_model import predict as restyle

output_file_path = sys.argv[1]
scheme = sys.argv[2]

N_LINES = 7
titles = [n['title'] for n in NEWS]
# temperatures = [5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.01, 0.0001]
temperatures = [0.0001, 0.01, 0.1, 0.25, 0.5, 1., 2., 5.]


def main(scheme):
    with open(output_file_path, 'w') as f:
        for t in tqdm(temperatures):
            start_time = time.time()

            if scheme == 'just-on-titles':
                dialogs = generate_on_titles(t)
            elif scheme == 'from-golds':
                dialogs = generate_from_golds(t)
            elif scheme == 'with-style':
                dialogs = generate_on_titles(t)
            else:
                raise NotImplementedError

            elapsed = time.time() - start_time

            f.write('\n\n====== Temperature: {}. (Took seconds: {:.03f}) ======\n\n'.format(t, elapsed))

            for dialog in dialogs:
                for line in dialog:
                    assert type(line) is str
                    f.write(line + '\n')
                f.write('---------------------------------\n')

            # print('Dialogs', dialogs)

            if scheme == 'with-style':
                restyled = [restyle(d) for d in dialogs]

                f.write('\n <= RESTYLINGS => \n')

                for dialog in restyled:
                    for line in dialog:
                        assert type(line) is str
                        f.write(line + '\n')
                    f.write('---------------------------------\n')

            f.flush()


def generate_on_titles(t):
    dialogs = generate_dialog(titles, N_LINES, t)
    dialogs = [[l['text'] for l in d] for d in dialogs]

    return dialogs


def generate_from_golds(t):
    dialogs = []

    for news in NEWS:
        article = [s + '.' for p in news['text'].splitlines() for s in p.split('. ')]
        article[-1] = article[-1][:-1] # Removing last dot from double dot
        dialog = [article[0]]

        for i in range(1, len(article)):
            condition = '|'.join(article[:i])
            predicted = generate_dialog([condition], 1, t)
            next_line = predicted[0][-1]['text']
            dialog.append(next_line)

        dialogs.append(dialog)

    return dialogs


if __name__ == '__main__':
    main(sys.argv[2])

