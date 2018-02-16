import argparse

from utils.data_utils import tokenize


def main(command, args):
    "Controls the project"
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run project commands')
    subparsers = parser.add_subparsers(dest='command')

    mask_parser = subparsers.add_parser('apply_mask')
    mask_parser.add_argument('--mask')
    mask_parser.add_argument('--img_in')
    mask_parser.add_argument('--img_out')

    merge_parser = subparsers.add_parser('merge_predictions')
    merge_parser.add_argument('--mask')
    merge_parser.add_argument('--img_out')

    # TODO(universome): finish this
    train_parser = subparsers.add_parser('train')

    args = parser.parse_args()

    if args.command == 'train':
        print('Well... train cmd is not available yey')

    else:
        print('No command is provided')
