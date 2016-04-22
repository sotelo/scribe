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
from utils import char2code, sample_parse, full_plot

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

scribe = Scribe(
    k=saved_args.num_mixture,
    rec_h_dim=saved_args.rnn_size,
    att_size=saved_args.size_attention,
    num_letters=saved_args.num_letters,
    sampling_bias=args.sampling_bias,
    attention_type=saved_args.attention_type,
    attention_alignment=saved_args.attention_alignment,
    name="scribe")

data, data_mask, context, context_mask, start_flag = \
    scribe.symbolic_input_variables()

sample_x, sample_pi, sample_phi, sample_pi_att, updates_sample = \
    scribe.sample_model(
        context, context_mask, args.num_steps, args.num_samples)

model = Model(sample_x)
model.set_parameter_values(parameters)

phrase = args.phrase + "  "
phrase = [char2code[char_] for char_ in phrase]
phrase = numpy.array(phrase, dtype='int32').reshape([-1, 1])
phrase = numpy.repeat(phrase, args.num_samples, axis=1).T
phrase_mask = numpy.ones(phrase.shape, dtype='float32')

tf = function(
    [context, context_mask],
    [sample_x, sample_pi, sample_phi, sample_pi_att],
    updates=updates_sample)

sampled_values = tf(phrase, phrase_mask)
sampled_values = [smpl.swapaxes(0, 1) for smpl in sampled_values]
sampled_values, sampled_pi, sampled_phi, sampled_pi_att = sampled_values

for sample in range(args.num_samples):
    # Heuristic for deciding when to end the sampling.
    phi = sampled_phi[sample]
    try:
        idx = numpy.where((
            phi[:, -1, numpy.newaxis] > phi[:, :-1]).all(axis=1))[0][0]
    except:
        print "Its better to increase the number of samples."

        idx = args.num_steps

    full_plot(
        sampled_values[sample, :idx],
        sampled_pi[sample, :idx],
        sampled_phi[sample, :idx],
        sampled_pi_att[sample, :idx, :, 0],
        os.path.join(
            args.save_dir, 'samples',
            args.samples_name + str(sample) + ".png"))
