import os
from algorithms import Adasecant
from blocks import initialization
from blocks.algorithms import (
    Adam, CompositeRule, GradientDescent, StepClipping)
from blocks.extensions import (Printing, Timing)
from blocks.extensions.monitoring import (
    DataStreamMonitoring, TrainingDataMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.saveload import Checkpoint, Load
from blocks.extensions.training import TrackTheBest
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
import cPickle
from extensions import LearningRateSchedule, Plot, TimedFinish, Write
from iam_on_line import stream_handwriting
from model import Scribe
from theano import function
from utils import train_parse

parser = train_parse()
args = parser.parse_args()

if args.algorithm == "adasecant":
    args.lr_schedule = False

rec_h_dim = args.rnn_size
att_size = args.size_attention
k = args.num_mixture
exp_name = args.experiment_name
save_dir = args.save_dir

print "Saving config ..."
with open(os.path.join(save_dir, 'config', exp_name + '.pkl'), 'w') as f:
    cPickle.dump(args, f)
print "Finished saving."

w_init = initialization.IsotropicGaussian(0.01)
b_init = initialization.Constant(0.)

train_stream = stream_handwriting(
    ('train',), args.batch_size, args.train_seq_length, args.num_letters)

valid_stream = stream_handwriting(
    ('valid',), args.batch_size, args.valid_seq_length, args.num_letters, 5)

x_tr, x_mask_tr, context_tr, context_mask_tr, flag_tr = \
    next(train_stream.get_epoch_iterator())

scribe = Scribe(
    k=args.num_mixture,
    rec_h_dim=args.rnn_size,
    att_size=args.size_attention,
    num_letters=args.num_letters,
    sampling_bias=0.,
    attention_type=args.attention_type,
    attention_alignment=args.attention_alignment,
    weights_init=w_init,
    biases_init=b_init,
    name="scribe")
scribe.initialize()

data, data_mask, context, context_mask, start_flag = \
    scribe.symbolic_input_variables()

cost, extra_updates = scribe.compute_cost(
    data, data_mask, context, context_mask, start_flag, args.batch_size)
cost.name = 'nll'

valid_cost = cost

cg = ComputationGraph(cost)
model = Model(cost)
parameters = cg.parameters

if args.algorithm == "adam":
    step_rule = CompositeRule([StepClipping(10.), Adam(args.learning_rate)])
elif args.algorithm == "adasecant":
    step_rule = Adasecant()

algorithm = GradientDescent(
    cost=cost,
    parameters=parameters,
    step_rule=step_rule,
    on_unused_sources='warn')
algorithm.add_updates(extra_updates)

monitoring_vars = [cost]

if args.lr_schedule:
    lr = algorithm.step_rule.components[1].learning_rate
    monitoring_vars.append(lr)

train_monitor = TrainingDataMonitoring(
    variables=monitoring_vars,
    every_n_batches=args.save_every,
    after_epoch=False,
    prefix="train")

valid_monitor = DataStreamMonitoring(
    [valid_cost],
    valid_stream,
    every_n_batches=args.save_every,
    after_epoch=False,
    prefix="valid")

# Multi GPU
worker = None
if args.platoon_port:
    from blocks_extras.extensions.synchronization import (
        Synchronize, SynchronizeWorker)
    from platoon.param_sync import ASGD

    sync_rule = ASGD()
    worker = SynchronizeWorker(
        sync_rule, control_port=args.platoon_port, socket_timeout=2000)

extensions = []

if args.load_experiment and (not worker or worker.is_main_worker):
    extensions += [Load(os.path.join(
        save_dir, "pkl", "best_" + args.load_experiment + ".tar"))]

extensions += [
    Timing(every_n_batches=args.save_every),
    train_monitor]

if not worker or worker.is_main_worker:
    extensions += [
        valid_monitor,
        TrackTheBest(
            'valid_nll',
            every_n_batches=args.save_every,
            before_first_epoch=True),
        Plot(
            os.path.join(save_dir, "progress", exp_name + ".png"),
            [['train_nll', 'valid_nll']],
            every_n_batches=args.save_every,
            email=False),
        Checkpoint(
            os.path.join(save_dir, "pkl", "best_" + exp_name + ".tar"),
            after_training=False,
            save_separately=['log'],
            use_cpickle=True,
            save_main_loop=False)
        .add_condition(
            ["after_batch", "before_training"],
            predicate=OnLogRecord('valid_nll_best_so_far'))]

    if args.lr_schedule:
        extensions += [
            LearningRateSchedule(
                lr, 'valid_nll',
                os.path.join(save_dir, "pkl", "best_" + exp_name + ".tar"),
                patience=10,
                num_cuts=5,
                every_n_batches=args.save_every)]

if args.plot_every:
    sample_x, sample_pi, sample_phi, sample_pi_att, updates_sample = \
        scribe.sample_model(
            context, context_mask, args.num_steps, args.num_samples)
    sampling_function = function(
        [context, context_mask], sample_x, updates=updates_sample)

    extensions += [
        Write(
            sampling_function,
            every_n_batches=args.plot_every,
            n_samples=args.num_samples,
            save_name=os.path.join(save_dir, "samples", exp_name))]

if worker:
    extensions += [
        Synchronize(
            worker,
            after_batch=True,
            before_epoch=True)]

extensions += [
    Printing(
        after_epoch=False,
        every_n_batches=args.save_every)]

if args.time_limit:
    extensions += [TimedFinish(args.time_limit)]

main_loop = MainLoop(
    model=model,
    data_stream=train_stream,
    algorithm=algorithm,
    extensions=extensions)

print "Training starting."
main_loop.run()
