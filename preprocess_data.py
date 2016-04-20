"""Preprocess handwriting dataset.

This code should only be runned once.
"""
import os
import re
import collections
import h5py
import tarfile
import numpy

from lxml import etree
from fuel.datasets.hdf5 import H5PYDataset
from utils import char2code, unk_char

data_path = os.environ['FUEL_DATA_PATH']
data_path = os.path.join(data_path, 'handwriting/')

input_file = os.path.join(data_path, 'lineStrokes-all.tar.gz')

file_name = "handwriting.hdf5"
hdf5_path = os.path.join(data_path, file_name)

raw_data = tarfile.open(input_file)

transcript_files = []
strokes = []
idx = 0
for member in raw_data.getmembers():
    if member.isreg():
        transcript_files.append(member.name)
        content = raw_data.extractfile(member)
        tree = etree.parse(content)
        root = tree.getroot()
        content.close()
        points = []
        for StrokeSet in root:
            for i, Stroke in enumerate(StrokeSet):
                for Point in Stroke:
                    points.append([
                        i,
                        int(Point.attrib['x']),
                        int(Point.attrib['y'])])
        points = numpy.array(points)
        points[:, 2] = -points[:, 2]
        change_stroke = points[:-1, 0] != points[1:, 0]
        pen_up = points[:, 0] * 0
        pen_up[:-1][change_stroke] = 1
        pen_up[-1] = 1
        points[:, 0] = pen_up
        strokes.append(points)
        idx += 1

strokes_bp = strokes

strokes = [x[1:] - x[:-1] for x in strokes]
strokes = [numpy.vstack([[0, 0, 0], x]) for x in strokes]

for i, stroke in enumerate(strokes):
    strokes[i][:, 0] = strokes_bp[i][:, 0]

transcript_files = [x.split("/")[-1] for x in transcript_files]
transcript_files = [re.sub('-[0-9][0-9].xml', '.txt', x)
                    for x in transcript_files]

counter = collections.Counter(transcript_files)

#######################
# OBTAIN TRANSCRIPTS
#######################

input_file = os.path.join(data_path, 'ascii-all.tar.gz')

raw_data = tarfile.open(input_file)
member = raw_data.getmembers()[10]

# This code was written by Kyle Kastner: @kastnerkyle
all_transcripts = []
for member in raw_data.getmembers():
    if member.isreg() and member.name.split("/")[-1] in transcript_files:
        fp = raw_data.extractfile(member)

        cleaned = [t.strip() for t in fp.readlines()
                   if t != '\r\n' and
                   t != '\n' and
                   t != '\r\n' and
                   t.strip() != '']

        # Try using CSR
        idx = [n for n, li in enumerate(cleaned) if li == "CSR:"][0]
        cleaned_sub = cleaned[idx + 1:]
        corrected_sub = []
        for li in cleaned_sub:
            # Handle edge case with %%%%% meaning new line?
            if "%" in li:
                li2 = re.sub('\%\%+', '%', li).split("%")
                li2 = [l.strip() for l in li2]
                corrected_sub.extend(li2)
            else:
                corrected_sub.append(li)

        if counter[member.name.split("/")[-1]] != len(corrected_sub):
            pass

        all_transcripts.extend(corrected_sub)

# Last file transcripts are almost garbage
all_transcripts[-1] = 'A move to stop'
all_transcripts.append('garbage')
all_transcripts.append('A move to stop')
all_transcripts.append('garbage')
all_transcripts.append('A move to stop')
all_transcripts.append('A move to stop')
all_transcripts.append('Marcus Luvki')
all_transcripts.append('Hallo Well')

####################
# Filter and Shuffle
####################

# Remove outliers and big / small sequences
# Makes a BIG difference.
filter_ = [len(x) <= 1200 and len(x) >= 301 and
           x.max() <= 2000 and x.min() >= -1000 for x in strokes]

strokes = [x for x, cond in zip(strokes, filter_) if cond]
all_transcripts = [x for x, cond in zip(all_transcripts, filter_) if cond]


# Computing mean and variance seems to not be necessary.
# Training is going slower than just scaling.

# Remove outliers

all_strokes = numpy.vstack(strokes)

data_mean = all_strokes.mean(axis=0)
data_std = all_strokes.std(axis=0)

data_mean[0] = 0.
data_std[0] = 1.

strokes = [(x - data_mean) / data_std for x in strokes]

num_examples = len(strokes)

# Shuffle for train/validation/test division
shuffle_idx = numpy.random.permutation(num_examples)

strokes = [strokes[x] for x in shuffle_idx]
all_transcripts = [all_transcripts[x] for x in shuffle_idx]

###################
# Create HDF5 File:
###################

num_files = len(strokes)
train_examples = int(num_examples * 0.97)

h5file = h5py.File(hdf5_path, mode='w')

features = h5file.create_dataset(
    'features', (num_examples,),
    dtype=h5py.special_dtype(vlen=numpy.dtype('float32')))

features_shapes = h5file.create_dataset(
    'features_shapes', (num_examples, 2), dtype='int32')

features.dims.create_scale(features_shapes, 'shapes')
features.dims[0].attach_scale(features_shapes)

features_shape_labels = h5file.create_dataset(
    'features_shape_labels', (2,), dtype='S7')
features_shape_labels[...] = [
    'time_step'.encode('utf8'),
    'feature_type'.encode('utf8')]
features.dims.create_scale(
    features_shape_labels, 'shape_labels')
features.dims[0].attach_scale(features_shape_labels)

transcripts = h5file.create_dataset(
    'transcripts', (num_examples,),
    dtype=h5py.special_dtype(vlen=numpy.dtype('int16')))

features[...] = [x.flatten() for x in strokes]
features_shapes[...] = [numpy.array(x.shape) for x in strokes]
transcripts[...] = [numpy.array([char2code.get(x, char2code[unk_char])
                    for x in transcript]) for transcript in all_transcripts]

split_dict = {
    'train': {'features': (0, train_examples),
              'transcripts': (0, train_examples)},
    'valid': {'features': (train_examples, num_examples),
              'transcripts': (train_examples, num_examples)}}

h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

h5file.flush()
h5file.close()
numpy.savez(
    os.path.join(data_path, 'handwriting_std.npz'),
    data_mean=data_mean, data_std=data_std)
