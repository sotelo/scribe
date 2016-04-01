"""Sampling code for the scribe.

Loads the trained model and samples.
"""

import numpy
import os
import cPickle
import logging

from blocks.serialization import load_parameters
from blocks.model import Model
from model import Scribe
from theano import function
from utils import char2code, sample_parse

logging.basicConfig()

parser = sample_parse()
args = parser.parse_args()

with open(os.path.join(
        args.save_dir, 'config',
        args.experiment_name + '.pkl')) as f:
    saved_args = cPickle.load(f)

with open(os.path.join(
        args.save_dir, "pkl",
        "best_" + args.experiment_name + ".tar"), 'rb') as src:
    parameters = load_parameters(src)

parameters = {k.replace('scribe/', ''): v for k, v in parameters.items()}

scribe = Scribe(
    k=saved_args.num_mixture,
    rec_h_dim=saved_args.rnn_size,
    att_size=saved_args.size_attention,
    num_letters=saved_args.num_letters,
    sampling_bias=args.sampling_bias)

data, data_mask, context, context_mask, start_flag = \
    scribe.symbolic_input_variables()

sample_x, updates_sample = scribe.sample_model(
    context, context_mask, args.num_steps, args.num_samples)

model = Model(sample_x)

model.set_parameter_values(parameters)

phrase = args.phrase + "  "
phrase = [char2code[char_] for char_ in phrase]
phrase = numpy.array(phrase, dtype='int32').reshape([-1, 1])
phrase = numpy.repeat(phrase, args.num_samples, axis=1).T

phrase_mask = numpy.ones(phrase.shape, dtype='float32')

sampled_values = function(
    [context, context_mask],
    sample_x,
    updates=updates_sample)(phrase, phrase_mask)

print sampled_values.shape
