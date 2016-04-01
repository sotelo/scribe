import matplotlib
import numpy

from blocks.extensions import SimpleExtension
from pandas import DataFrame
from utils import char2code, plot_tight

matplotlib.use('Agg')
from matplotlib import pyplot


class Plot(SimpleExtension):
    """Alternative plot extension for blocks.

    Parameters
    ----------
    document : str
        The name of the plot file. Use a different name for each
        experiment if you are storing your plots.
    channels : list of lists of strings
        The names of the monitor channels that you want to plot. The
        channels in a single sublist will be plotted together in a single
        figure, so use e.g. ``[['test_cost', 'train_cost'],
        ['weight_norms']]`` to plot a single figure with the training and
        test cost, and a second figure for the weight norms.
    """

    # Tableau 10 colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self, document, channels, email=True, **kwargs):
        self.plots = {}
        self.document = document
        self.num_plots = len(channels)
        self.channels = channels
        self.all_channels = list(set([x for small in channels for x in small]))
        self.document = document
        super(Plot, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        log = self.main_loop.log
        df = DataFrame.from_dict(log, orient='index')
        df = df[self.all_channels].astype(float)
        df = df.interpolate('index')

        fig, axarr = pyplot.subplots(self.num_plots, sharex=True)

        if self.num_plots > 1:
            for i, channel in enumerate(self.channels):
                df[channel].plot(ax=axarr[i])
        else:
            df[self.channels[0]].plot()

        pyplot.savefig(self.document)
        pyplot.close()


class Write(SimpleExtension):
    """Make your scribe write.

    Parameters
    ----------
    steps : int Number of points to generate
    """

    def __init__(
            self,
            sample_x,
            steps=300,
            n_samples=5,
            phrase="hello my friend how are you doing today",
            save_name="sample_scribe", **kwargs):
        super(Write, self).__init__(**kwargs)
        phrase = phrase + "  "
        phrase = [char2code[char_] for char_ in phrase]
        phrase = numpy.array(phrase, dtype='int32').reshape([-1, 1])
        phrase = numpy.repeat(phrase, n_samples, axis=1).T

        self.tf = sample_x
        self.phrase = phrase
        self.save_name = save_name
        self.step = 0

    def do(self, callback_name, *args):
        samples = self.tf(
            self.phrase,
            numpy.ones(self.phrase.shape, dtype='float32')).swapaxes(0, 1)[:5]

        for i, sample in enumerate(samples):
            plot_tight(
                sample,
                self.save_name + "_" + str(self.step) + "_" + str(i) + ".png")

        self.step += 1
