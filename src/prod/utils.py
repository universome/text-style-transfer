import os


def create_get_path_fn(models_dir):
    versions = set([extract_iter_num(f) for f in os.listdir(models_dir) if '-' in f])
    latest = max(versions)
    get_path = lambda m, ext='pth': os.path.join(models_dir, '{}-{}.{}'.format(m, latest, ext))

    print('Latest iter (version) found: {}.'.format(latest))

    return get_path


def extract_iter_num(filename):
    return int(filename.split('-')[1].split('.')[0])


def validate(data, property_name, property_type):
    # TODO: why on hell do you check types manually?!
    if not property_name in data:
        return 'Data should inlcude `{}`'.format(property_name)

    if not type(data[property_name]) is property_type:
        return 'Property `{}` should have type `{}`'.format(property_name, property_type)

    return None
