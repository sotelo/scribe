from blocks.bricks import Initializable, Linear, Random
from blocks.bricks.base import application
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent
from blocks.bricks.sequence_generators import AbstractEmitter
from blocks.roles import add_role, INITIAL_STATE
from blocks.utils import shared_floatx_zeros

import numpy

import theano
from theano import tensor, shared

floatX = theano.config.floatX


# https://gist.github.com/benanne/2300591
def one_hot(t, r=None):
    """Compute one hot encoding.

    given a tensor t of dimension d with integer values from range(r), return a
    new tensor of dimension d + 1 with values 0/1, where the last dimension
    gives a one-hot representation of the values in t.

    if r is not given, r is set to max(t) + 1
    """
    if r is None:
        r = tensor.max(t) + 1

    ranges = tensor.shape_padleft(tensor.arange(r), t.ndim)
    return tensor.eq(ranges, tensor.shape_padright(t, 1))


def logsumexp(x, axis=None):
    x_max = tensor.max(x, axis=axis, keepdims=True)
    z = tensor.log(
        tensor.sum(tensor.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return z.sum(axis=axis)


def predict(probs, axis=-1):
    return tensor.argmax(probs, axis=axis)


def bivariate_gmm(y, mu, sigma, corr, coeff, binary, epsilon=1e-5):
    """Bivariate gaussian mixture model negative log-likelihood.

    Parameters

    """
    n_dim = y.ndim
    shape_y = y.shape
    y = y.reshape((-1, shape_y[-1]))
    y = tensor.shape_padright(y)

    data_pen = y[:, 0, :]
    data_x = y[:, 1, :]
    data_y = y[:, 2, :]

    sigma_x = sigma[:, 0, :]
    sigma_y = sigma[:, 1, :]

    std_e_x = (data_x - mu[:, 0, :]) / sigma_x
    std_e_y = (data_y - mu[:, 1, :]) / sigma_y

    binary = (binary + epsilon) * (1. - 2. * epsilon)

    c_b = tensor.sum(
        tensor.xlogx.xlogy0(data_pen, binary) +
        tensor.xlogx.xlogy0(1. - data_pen, 1. - binary), axis=1)

    buff = 1. - corr**2 + epsilon

    z = std_e_x**2 + std_e_y**2 - 2. * corr * std_e_x * std_e_y

    cost = - z / (2. * buff) - 0.5 * tensor.log(buff) - \
        tensor.log(sigma_x) - tensor.log(sigma_y) - tensor.log(2. * numpy.pi)

    nll = -logsumexp(tensor.log(coeff) + cost, axis=1) - c_b

    return nll.reshape(shape_y[:-1], ndim=n_dim - 1)


class BivariateGMMEmitter(AbstractEmitter, Initializable, Random):
    """A mixture of gaussians emitter for x,y and logistic for pen-up/down.

    Parameters
    ----------
    k : number of components

    """

    def __init__(self, k=20, epsilon=1e-5, sampling_bias=0., **kwargs):
        self.k = k
        self.epsilon = epsilon
        self.sampling_bias = sampling_bias
        super(BivariateGMMEmitter, self).__init__(**kwargs)

    @application(outputs=["mu", "sigma", "corr", "coeff", "penup"])
    def components(self, readouts):
        """Extract parameters of the distribution."""
        k = self.k
        readouts = readouts.reshape((-1, self.get_dim('inputs')))

        # Reshaped
        mu = readouts[:, 0:2 * k].reshape((-1, 2, k))
        sigma = readouts[:, 2 * k:4 * k].reshape((-1, 2, k))
        corr = readouts[:, 4 * k:5 * k]
        weight = readouts[:, 5 * k:6 * k]
        penup = readouts[:, 6 * k:]

        sigma = tensor.exp(sigma - self.sampling_bias) + self.epsilon
        # sigma = tensor.nnet.softplus(sigma - self.sampling_bias) + \
        #     self.epsilon
        corr = tensor.tanh(corr)
        weight = tensor.nnet.softmax(weight * (1. + self.sampling_bias)) + \
            self.epsilon
        penup = tensor.nnet.sigmoid(penup * (1. + self.sampling_bias))

        mu.name = "mu"
        sigma.name = "sigma"
        corr.name = "corr"
        weight.name = "coeff"
        penup.name = "penup"

        return mu, sigma, corr, weight, penup

    @application
    def emit(self, readouts):
        """Sample from the distribution.

        Parameters:
            readouts: readouts from the rnn + attention

        """
        mu, sigma, corr, coeff, penup = self.components(readouts)

        idx = predict(
            self.theano_rng.multinomial(
                pvals=coeff,
                dtype=coeff.dtype
            ), axis=1)

        mu = mu[tensor.arange(mu.shape[0]), :, idx]
        sigma = sigma[tensor.arange(sigma.shape[0]), :, idx]
        corr = corr[tensor.arange(corr.shape[0]), idx]

        mu_x = mu[:, 0]
        mu_y = mu[:, 1]
        sigma_x = sigma[:, 0]
        sigma_y = sigma[:, 1]

        z = self.theano_rng.normal(
            size=mu.shape, avg=0., std=1., dtype=mu.dtype)

        un = self.theano_rng.uniform(size=penup.shape)
        penup = tensor.cast(un < penup, floatX)

        s_x = tensor.shape_padright(mu_x + sigma_x * z[:, 0])
        s_y = mu_y + sigma_y * ((z[:, 0] * corr) + (
            z[:, 1] * tensor.sqrt(1. - corr**2)))
        s_y = tensor.shape_padright(s_y)
        s = tensor.concatenate([penup, s_x, s_y], axis=1)

        return s

    @application
    def cost(self, readouts, outputs):
        """Bivariate Gaussian NLL.

        Parameters:
            readouts: readouts from the transition + attention
            outputs: true value from the data

        """
        mu, sigma, corr, coeff, penup = self.components(readouts)
        return bivariate_gmm(
            outputs, mu, sigma, corr, coeff, penup, self.epsilon)

    @application
    def initial_outputs(self, batch_size):
        return tensor.zeros((batch_size, 3))

    def get_dim(self, name):
        if name == 'inputs':
            # (2k: mean, 2k: variance, k: corr, k: weights, 1:penup)
            return 6 * self.k + 1
        if name == 'outputs':
            return 3
        return super(BivariateGMMEmitter, self).get_dim(name)


class Scribe(Initializable):
    def __init__(
            self,
            k=20,
            rec_h_dim=400,
            att_size=10,
            num_letters=68,
            sampling_bias=0.,
            attention_type="graves",
            epsilon=1e-6,
            attention_alignment=1.,
            **kwargs):
        super(Scribe, self).__init__(**kwargs)

        # For now only softmax and graves are supported.
        assert attention_type in ["graves", "softmax"]

        readouts_dim = 1 + 6 * k

        self.k = k
        self.rec_h_dim = rec_h_dim
        self.att_size = att_size
        self.num_letters = num_letters
        self.sampling_bias = sampling_bias
        self.attention_type = attention_type
        self.epsilon = epsilon
        self.attention_alignment = attention_alignment

        self.cell1 = GatedRecurrent(dim=rec_h_dim, name='cell1')

        self.inp_to_h1 = Fork(
            output_names=['cell1_inputs', 'cell1_gates'],
            input_dim=3,
            output_dims=[rec_h_dim, 2 * rec_h_dim],
            name='inp_to_h1')

        self.h1_to_readout = Linear(
            input_dim=rec_h_dim,
            output_dim=readouts_dim,
            name='h1_to_readout')

        self.h1_to_att = Fork(
            output_names=['alpha', 'beta', 'kappa'],
            input_dim=rec_h_dim,
            output_dims=[att_size] * 3,
            name='h1_to_att')

        self.att_to_h1 = Fork(
            output_names=['cell1_inputs', 'cell1_gates'],
            input_dim=num_letters,
            output_dims=[rec_h_dim, 2 * rec_h_dim],
            name='att_to_h1')

        self.att_to_readout = Linear(
            input_dim=num_letters,
            output_dim=readouts_dim,
            name='att_to_readout')

        self.emitter = BivariateGMMEmitter(
            k=k,
            sampling_bias=sampling_bias)

        self.children = [
            self.cell1,
            self.inp_to_h1,
            self.h1_to_readout,
            self.h1_to_att,
            self.att_to_h1,
            self.att_to_readout,
            self.emitter]

    def _allocate(self):
        self.initial_w = shared_floatx_zeros(
            (self.num_letters,), name="initial_w")

        add_role(self.initial_w, INITIAL_STATE)

    def symbolic_input_variables(self):
        data = tensor.tensor3('features')
        data_mask = tensor.matrix('features_mask')
        context = tensor.imatrix('transcripts')
        context_mask = tensor.matrix('transcripts_mask')
        start_flag = tensor.scalar('start_flag')

        return data, data_mask, context, context_mask, start_flag

    def initial_states(self, batch_size):
        initial_h1 = self.cell1.initial_states(batch_size)
        initial_kappa = shared_floatx_zeros((batch_size, self.att_size))
        initial_w = tensor.repeat(self.initial_w[None, :], batch_size, 0)
        last_h1 = shared_floatx_zeros((batch_size, self.rec_h_dim))
        last_w = shared_floatx_zeros((batch_size, self.num_letters))
        use_last_states = shared(numpy.asarray(0., dtype=floatX))

        return initial_h1, initial_kappa, initial_w, \
            last_h1, last_w, use_last_states

    @application
    def compute_cost(
            self,
            data,
            data_mask,
            context,
            context_mask,
            start_flag,
            batch_size):
        x = data[:-1]
        target = data[1:]
        mask = data_mask[1:]
        xinp_h1, xgat_h1 = self.inp_to_h1.apply(x)
        context_oh = one_hot(context, self.num_letters) * \
            tensor.shape_padright(context_mask)

        initial_h1, initial_kappa, initial_w, \
            last_h1, last_w, use_last_states = \
            self.initial_states(batch_size)

        input_h1 = tensor.switch(
            use_last_states, last_h1, initial_h1)
        input_w = tensor.switch(
            use_last_states, last_w, initial_w)

        u = tensor.shape_padleft(
            tensor.arange(context.shape[1], dtype=floatX), 2)

        def step(xinp_h1_t, xgat_h1_t, h1_tm1, k_tm1, w_tm1, ctx):

            attinp_h1, attgat_h1 = self.att_to_h1.apply(w_tm1)

            h1_t = self.cell1.apply(
                xinp_h1_t + attinp_h1,
                xgat_h1_t + attgat_h1, h1_tm1, iterate=False)

            a_t, b_t, k_t = self.h1_to_att.apply(h1_t)

            if self.attention_type == "softmax":
                a_t = tensor.nnet.softmax(a_t)
            else:
                a_t = tensor.exp(a_t)

            b_t = tensor.exp(b_t) + self.epsilon
            k_t = k_tm1 + self.attention_alignment * tensor.exp(k_t)

            a_t = tensor.shape_padright(a_t)
            b_t = tensor.shape_padright(b_t)
            k_t_ = tensor.shape_padright(k_t)

            # batch size X att size X len context
            if self.attention_type == "softmax":
                # numpy.sqrt(1/(2*numpy.pi)) is the weird number
                phi_t = 0.3989422917366028 * tensor.sum(
                    a_t * tensor.sqrt(b_t) *
                    tensor.exp(-0.5 * b_t * (k_t_ - u)**2), axis=1)
            else:
                phi_t = tensor.sum(
                    a_t * tensor.exp(-b_t * (k_t_ - u)**2), axis=1)

            # batch size X len context X num letters
            w_t = (tensor.shape_padright(phi_t) * ctx).sum(axis=1)

            return h1_t, k_t, w_t

        (h1, kappa, w), scan_updates = theano.scan(
            fn=step,
            sequences=[xinp_h1, xgat_h1],
            non_sequences=[context_oh],
            outputs_info=[input_h1, initial_kappa, input_w])

        readouts = self.h1_to_readout.apply(h1) + \
            self.att_to_readout.apply(w)

        cost = self.emitter.cost(readouts, target)
        cost = (cost * mask).sum() / (mask.sum() + 1e-5) + 0. * start_flag

        updates = []
        updates.append((last_h1, h1[-1]))
        updates.append((
            initial_kappa,
            tensor.switch(start_flag, 0. * initial_kappa, kappa[-1])))
        updates.append((last_w, w[-1]))
        updates.append((use_last_states, 1. - start_flag))

        return cost, scan_updates + updates

    @application
    def sample_model(self, context, context_mask, n_steps, batch_size):

        initial_h1, initial_kappa, initial_w, \
            last_h1, last_w, use_last_states = \
            self.initial_states(batch_size)

        initial_x = self.emitter.initial_outputs(batch_size)

        context_oh = one_hot(context, self.num_letters) * \
            tensor.shape_padright(context_mask)

        u = tensor.shape_padleft(
            tensor.arange(context.shape[1], dtype=floatX), 2)

        def sample_step(x_tm1, h1_tm1, k_tm1, w_tm1, ctx):
            xinp_h1_t, xgat_h1_t = self.inp_to_h1.apply(x_tm1)

            attinp_h1, attgat_h1 = self.att_to_h1.apply(w_tm1)

            h1_t = self.cell1.apply(
                xinp_h1_t + attinp_h1,
                xgat_h1_t + attgat_h1, h1_tm1, iterate=False)

            a_t, b_t, k_t = self.h1_to_att.apply(h1_t)

            if self.attention_type == "softmax":
                a_t = tensor.nnet.softmax(a_t)
            else:
                a_t = tensor.exp(a_t)

            b_t = tensor.exp(b_t) + self.epsilon
            k_t = k_tm1 + self.attention_alignment * tensor.exp(k_t)

            a_t = tensor.shape_padright(a_t)
            b_t = tensor.shape_padright(b_t)
            k_t_ = tensor.shape_padright(k_t)

            # batch size X att size X len context
            if self.attention_type == "softmax":
                # numpy.sqrt(1/(2*numpy.pi)) is the weird number
                phi_t = 0.3989422917366028 * tensor.sum(
                    a_t * tensor.sqrt(b_t) *
                    tensor.exp(-0.5 * b_t * (k_t_ - u)**2), axis=1)
            else:
                phi_t = tensor.sum(
                    a_t * tensor.exp(-b_t * (k_t_ - u)**2), axis=1)

            # batch size X len context X num letters
            w_t = (tensor.shape_padright(phi_t) * ctx).sum(axis=1)

            readout_t = self.h1_to_readout.apply(h1_t) + \
                self.att_to_readout.apply(w_t)

            x_t = self.emitter.emit(readout_t)

            mu_t, sigma_t, corr_t, pi_t, penup_t = \
                self.emitter.components(readout_t)

            return x_t, h1_t, k_t, w_t, pi_t, phi_t, a_t

        (sample_x, h1, k, w, pi, phi, pi_att), updates = theano.scan(

            fn=sample_step,
            n_steps=n_steps,
            sequences=[],
            non_sequences=[context_oh],
            outputs_info=[
                initial_x.eval(),
                initial_h1,
                initial_kappa,
                initial_w,
                None,
                None,
                None])

        return sample_x, pi, phi, pi_att, updates
