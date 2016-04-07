import argparse
import numpy
import os

save_dir = os.environ['RESULTS_DIR']
if 'handwriting' not in save_dir:
    save_dir = os.path.join(save_dir, 'handwriting/')

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

data_path = os.environ['FUEL_DATA_PATH']
data_path = os.path.join(data_path, 'handwriting/')
std_values = numpy.load(os.path.join(data_path, 'handwriting_std.npz'))
data_mean = std_values['data_mean']
data_std = std_values['data_std']

all_chars = ([chr(ord('a') + i) for i in range(26)] +
             [chr(ord('A') + i) for i in range(26)] +
             [chr(ord('0') + i) for i in range(10)] +
             [',', '.', ' ', '"', '<UNK>', "'"])

code2char = dict(enumerate(all_chars))
char2code = {v: k for k, v in code2char.items()}
unk_char = '<UNK>'


def plot_tight(data, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    #std_x = data_std * data + data_mean
    std_x = data

    x = numpy.cumsum(std_x[:, 1])
    y = numpy.cumsum(std_x[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = numpy.where(std_x[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=1.5)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print "Error building image!: " + save_name

    pyplot.close()


def train_parse():
    """Parser for training arguments.

    Save dir is by default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='baseline',
                        help='name of the experiment.')
    parser.add_argument('--rnn_size', type=int, default=400,
                        help='size of RNN hidden state')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--train_seq_length', type=int, default=300,
                        help='RNN sequence length')
    parser.add_argument('--valid_seq_length', type=int, default=1200,
                        help='RNN sequence length')
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num_mixture', type=int, default=20,
                        help='number of gaussian mixtures')
    parser.add_argument('--save_dir', type=str,
                        default=save_dir,
                        help='save dir directory')
    parser.add_argument('--size_attention', type=int, default=10,
                        help='number of normal components for attention')
    parser.add_argument('--num_letters', type=int, default=68,
                        help='size of dictionary')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='number of samples')
    parser.add_argument('--num_steps', type=int, default=1000,
                        help='maximum size of each sample')
    parser.add_argument('--time_limit', type=float, default=None,
                        help='time in hours that the model will run')
    parser.add_argument('--load_experiment', type=str,
                        default=None,
                        help='name of the experiment that will be loaded')
    parser.add_argument('--platoon_port', type=int,
                        default=None,
                        help='port where platoon server is running')
    return parser


def sample_parse():
    """Parser for sampling arguments.

    Save dir is by default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='baseline',
                        help='name of the experiment.')
    parser.add_argument('--sampling_bias', type=float, default=0.5,
                        help='the higher the bias the smoother the samples')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='number of samples')
    parser.add_argument('--num_steps', type=int, default=1000,
                        help='maximum size of each sample')
    parser.add_argument('--save_dir', type=str,
                        default='./trained/',
                        help='save dir directory')
    parser.add_argument('--phrase', type=str, default='what should i write',
                        help='phrase to write')
    return parser
