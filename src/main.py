import os
from os import path
import json
import argparse
import sys; sys.path.extend(['.', './src'])

from hasheroku import hasheroku

from src.runners import WordRecoveryRunner

def run_experiment(config_name):
    if config_name == 'word_recovery':
        config_path = path.join('experiments/configs', config_name + '.json')

        with open(config_path, encoding='utf-8') as config_file:
            config = json.load(config_file)

        results_dir = path.join('experiments/results', hasheroku(str(config)))

        if os.path.exists(results_dir):
            raise Exception('Directory {} already exists.'.format(results_dir))

        runner = WordRecoveryRunner(config, results_dir)
    else:
        raise NotImplemented

    runner.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run project commands')
    subparsers = parser.add_subparsers(dest='command')

    exp_parser = subparsers.add_parser('run-experiment')
    exp_parser.add_argument('--config')

    args = parser.parse_args()

    if args.command == 'run-experiment':
        run_experiment(args.config)
    else:
        raise NotImplemented
