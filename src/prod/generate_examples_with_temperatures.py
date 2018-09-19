#!/usr/bin/env python
import sys
import time

from tqdm import tqdm

from news import NEWS
from dialog_model import predict

output_file_path = sys.argv[1]

N_LINES = 7
titles = [n['title'] for n in NEWS]
temperatures = [0.01, 0.1, 0.25, 0.5, 1., 2., 5., 10.]

with open(output_file_path, 'w') as f:
    for t in tqdm(temperatures):
        start_time = time.time()
        dialogs = predict(titles, N_LINES, t)
        elapsed = time.time() - start_time

        f.write('====== Temperature: {}. (Took seconds: {:.03f}) ======\n'.format(t, elapsed))

        for dialog in dialogs:
            for s in dialog:
                f.write(s + '\n')
            f.write('---------------------------------\n')

        f.flush()
