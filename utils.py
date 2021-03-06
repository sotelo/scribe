import argparse
import numpy
import os

save_dir = os.environ['RESULTS_DIR']
if 'handwriting' not in save_dir:
    save_dir = os.path.join(save_dir, 'handwriting/')

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

data_path = os.environ['FUEL_DATA_PATH']
data_path = os.path.join(data_path, 'handwriting/')
std_values = numpy.load(os.path.join(data_path, 'handwriting_std.npz'))
data_mean = std_values['data_mean']
data_std = std_values['data_std']

all_chars = ([chr(ord('a') + i) for i in range(26)] +
             [chr(ord('A') + i) for i in range(26)] +
             [chr(ord('0') + i) for i in range(10)] +
             [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+',
              ',', '-', '.', '/', ':', ';', '?', '[', ']', '<UNK>'])

code2char = dict(enumerate(all_chars))
char2code = {v: k for k, v in code2char.items()}
unk_char = '<UNK>'


def plot_tight(data, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    std_x = data_std * data + data_mean
    # std_x = data

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


def full_plot(data, pi, phi, pi_at, save_name=None):
    # Plot a single example.
    f, (ax1, ax2, ax3, ax4) = pyplot.subplots(4, 1)

    std_x = data_std * data + data_mean
    # std_x = data

    x = numpy.cumsum(std_x[:, 1])
    y = numpy.cumsum(std_x[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 20.)

    cuts = numpy.where(std_x[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax1.plot(
            x[start:cut_value], y[start:cut_value], 'k-', linewidth=1.5)
        start = cut_value + 1
    ax1.axis('equal')
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)

    def plot_matrix(data, ax):
        im = ax.imshow(
            data.T, aspect='auto', origin='lower', interpolation='nearest')
        cax = make_axes_locatable(ax).append_axes("right", size="1%", pad=0.05)
        pyplot.colorbar(im, cax=cax)

    plot_matrix(phi, ax2)
    plot_matrix(pi_at, ax3)
    plot_matrix(pi, ax4)

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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--train_seq_length', type=int, default=1200,
                        help='RNN sequence length')
    parser.add_argument('--valid_seq_length', type=int, default=1200,
                        help='RNN sequence length')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--num_mixture', type=int, default=20,
                        help='number of gaussian mixtures')
    parser.add_argument('--save_dir', type=str,
                        default=save_dir,
                        help='save dir directory')
    parser.add_argument('--size_attention', type=int, default=10,
                        help='number of normal components for attention')
    parser.add_argument('--num_letters', type=int, default=len(all_chars),
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
    parser.add_argument('--plot_every', type=int,
                        default=None,
                        help='how often do you want to plot')
    parser.add_argument('--platoon_port', type=int,
                        default=None,
                        help='port where platoon server is running')
    parser.add_argument('--attention_type', type=str,
                        default='softmax',
                        help='graves or softmax')
    parser.add_argument('--attention_alignment', type=float,
                        default=0.05,
                        help='initial lengths of each attention step')
    parser.add_argument('--algorithm', type=str,
                        default='adam',
                        help='adam or adasecant')
    parser.add_argument('--grad_clip', type=float,
                        default=0.9,
                        help='how much to clip the gradients. for adam is 10x')
    parser.add_argument('--lr_schedule', type=bool,
                        default=False,
                        help='whether to use the learning rate schedule')
    parser.add_argument('--sort_mult', type=int,
                        default=20,
                        help='number of minibatches to sort')
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
    parser.add_argument('--samples_name', type=str, default='sample',
                        help='name to save the samples.')
    parser.add_argument('--save_dir', type=str,
                        default='./trained/',
                        help='save dir directory')
    parser.add_argument('--phrase', type=str, default='what should i write',
                        help='phrase to write')
    return parser
