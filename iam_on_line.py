"""Dataset information.

This file contains the different functions needed to create the data stream
that the model will use.
"""

import os

from fuel import config
from fuel.datasets import H5PYDataset
from fuel.schemes import ConstantScheme, ShuffledExampleScheme
from fuel.streams import DataStream
from fuel.transformers import (
    AgnosticSourcewiseTransformer, Batch, Mapping, Padding, SortMapping,
    Transformer, Unpack)

import numpy


def _transpose(data):
    return data.swapaxes(0, 1)


def _length(data):
    return len(data[0])


class SourceMapping(AgnosticSourcewiseTransformer):
    """Apply a function to a subset of sources.

    Similar to the Mapping transformer but for a subset of sources.
    It will apply the same function to each source.
    Parameters
    ----------
    mapping : callable
    """

    def __init__(self, data_stream, mapping, **kwargs):
        """Initialization.

        Parameters:
            data_stream: DataStream
            mapping: callable object
        """
        self.mapping = mapping
        if data_stream.axis_labels:
            kwargs.setdefault('axis_labels', data_stream.axis_labels.copy())
        super(SourceMapping, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)

    def transform_any_source(self, source_data, _):
        return numpy.asarray(self.mapping(source_data))


class SegmentSequence(Transformer):
    """Segments the sequences in a batch.

    This transformer is useful to do tbptt. All the sequences to segment
    should have the time dimension as their first dimension.
    Parameters
    ----------
    data_stream : instance of :class:`DataStream`
        The wrapped data stream.
    seq_size : int
        maximum size of the resulting sequences.
    which_sources : tuple of str, optional
        sequences to segment
    add_flag : bool, optional
        add a flag indicating the beginning of a new sequence.
    flag_name : str, optional
        name of the source for the flag
    min_size : int, optional
        smallest possible sequence length for the last cut
    return_last : bool, optional
        return the last cut of the sequence, which might be different size
    share_value : bool, optional
        every cut will share the first value with the last value of past cut
    """

    def __init__(self,
                 data_stream,
                 seq_size=100,
                 which_sources=None,
                 add_flag=False,
                 flag_name=None,
                 min_size=10,
                 return_last=True,
                 share_value=False,
                 **kwargs):

        super(SegmentSequence, self).__init__(
            data_stream=data_stream,
            produces_examples=data_stream.produces_examples,
            **kwargs)

        if which_sources is None:
            which_sources = data_stream.sources
        self.which_sources = which_sources

        self.seq_size = seq_size
        self.step = 0
        self.data = None
        self.len_data = None
        self.add_flag = add_flag
        self.min_size = min_size
        self.share_value = share_value

        if not return_last:
            self.min_size += self.seq_size

        if flag_name is None:
            flag_name = u"start_flag"

        self.flag_name = flag_name

    @property
    def sources(self):
        return self.data_stream.sources + ((self.flag_name,)
                                           if self.add_flag else ())

    def get_data(self, request=None):
        flag = 0

        if self.data is None:
            self.data = next(self.child_epoch_iterator)
            idx = self.sources.index(self.which_sources[0])
            self.len_data = self.data[idx].shape[0]

        segmented_data = list(self.data)

        for source in self.which_sources:
            idx = self.sources.index(source)
            # Segment data:
            segmented_data[idx] = self.data[idx][
                self.step:(self.step + self.seq_size)]

        self.step += self.seq_size

        if self.share_value:
            self.step -= 1

        if self.step + self.min_size >= self.len_data:
            self.data = None
            self.len_data = None
            self.step = 1
            flag = 1

        if self.add_flag:
            segmented_data.append(flag)

        return tuple(segmented_data)


class Handwriting(H5PYDataset):
    filename = 'handwriting.hdf5'

    def __init__(self, which_sets, **kwargs):
        super(Handwriting, self).__init__(
            self.data_path,
            which_sets,
            load_in_memory=True,
            **kwargs)

    @property
    def data_path(self):
        return os.path.join(config.data_path[0], 'handwriting', self.filename)


def stream_handwriting(
        which_sets,
        batch_size,
        seq_size,
        num_letters,
        sorting_mult=20):

    dataset = Handwriting(which_sets)
    sorting_size = batch_size * sorting_mult
    num_examples = sorting_size * (dataset.num_examples / sorting_size)

    data_stream = DataStream.default_stream(
        dataset,
        iteration_scheme=ShuffledExampleScheme(num_examples))

    # Sort by length of the data sequence.
    data_stream = Batch(data_stream,
                        iteration_scheme=ConstantScheme(sorting_size))
    data_stream = Mapping(data_stream, SortMapping(_length))
    data_stream = Unpack(data_stream)
    data_stream = Batch(data_stream,
                        iteration_scheme=ConstantScheme(batch_size))

    data_stream = Padding(data_stream)
    data_stream = SourceMapping(
        data_stream, _transpose, which_sources=('features', 'features_mask'))
    data_stream = SegmentSequence(
        data_stream,
        seq_size=seq_size + 1,
        share_value=True,
        return_last=True,
        which_sources=('features', 'features_mask'),
        add_flag=True)
    return data_stream

if __name__ == "__main__":
    data_stream = stream_handwriting(('train',), 64, 100, 69)
    x_tr = next(data_stream.get_epoch_iterator())
