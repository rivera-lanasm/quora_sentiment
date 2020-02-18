import argparse


setting_0 = {
    "exp_name": "example",
    "num_epochs": 10,
    "num_iter_per_epoch": 10,
    "learning_rate": 0.001,
    "batch_size": 16,
    "state_size": [784],
    "max_to_keep":5
  }


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


if __name__ == "__main__":

    x = get_args()
    print(x)

