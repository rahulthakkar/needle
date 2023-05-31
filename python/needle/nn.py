"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(init.kaiming_uniform(self.in_features, self.out_features, 
                          shape=(self.in_features, self.out_features), device=device, dtype=dtype, requires_grad=True))
        self.has_bias = bias
        if bias:
            self.bias = Parameter(ops.transpose(init.kaiming_uniform(self.out_features, 1, shape=(self.out_features, 1), 
                          device=device, dtype=dtype, requires_grad=True)))

    def forward(self, X: Tensor) -> Tensor:
        out = X @ self.weight
        if self.has_bias:
            out += ops.broadcast_to(self.bias, shape=out.shape)
        return out


class Flatten(Module):
    def forward(self, X):
        from operator import mul
        from functools import reduce
        flattened_shape = (X.shape[0], reduce(mul, X.shape[1:]))
        return ops.reshape(X, flattened_shape)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return ops.sigmoid(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        batch_size, num_labels = logits.shape
        y_one_hot = init.one_hot(num_labels, y, device=logits.device, dtype=logits.dtype)
        # TODO: allow -ve axis
        Z_y = ops.summation(logits * y_one_hot, axes=(1,))
        return ops.summation(ops.logsumexp(logits, axes=(1,)) - Z_y) / batch_size


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _ = x.shape
        apply_broadcast = lambda param: ops.broadcast_squeezed(param, x.shape, (0,))
        if self.training:
            mean_x = ops.summation(x, axes=(0,)) / batch_size
            centered_x = x - apply_broadcast(mean_x)
            var_x = ops.summation(centered_x ** 2, axes=(0,)) / batch_size
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean_x.data
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * var_x.data
            weight, bias, var_x = map(apply_broadcast, (self.weight, self.bias, var_x))
            return weight * centered_x / ((var_x + self.eps) ** (1 / 2)) + bias
        else:
            weight, bias = map(apply_broadcast, (self.weight.data, self.bias.data))
            running_mean, running_var = map(apply_broadcast, (self.running_mean, self.running_var))
            return weight * (x - running_mean) / ((running_var + self.eps) ** (1/2)) + bias


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        axes = tuple(range(1, len(x.shape)))
        apply_broadcast_agg = lambda param: ops.broadcast_squeezed(param, x.shape, axes)
        apply_broadcast_params = lambda param: ops.broadcast_squeezed(param, x.shape, (0,))
        mean_x = apply_broadcast_agg(ops.summation(x, axes=axes) / self.dim)
        centered_x = x - mean_x
        var_x = apply_broadcast_agg(ops.summation(centered_x ** 2, axes=axes) / self.dim)
        weight, bias = map(apply_broadcast_params, (self.weight, self.bias))
        return weight * centered_x / ((var_x + self.eps) ** (1 / 2)) + bias


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p, device=x.device, dtype="bool", requires_grad=False)
            x = (x * mask) / (1-self.p)
        return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.has_bias = bias
        self.padding = (kernel_size - 1) // 2 ## assuming odd kernel size
        self.weight = Parameter(init.kaiming_uniform(in_channels * kernel_size**2, out_channels * kernel_size**2, 
                                  shape=(kernel_size, kernel_size, in_channels, out_channels), 
                                  device=device, dtype=dtype, requires_grad=True))
        if bias:
            limit = 1 / (in_channels * kernel_size**2)**(1/2)
            self.bias = Parameter(init.rand(out_channels, low=-limit, high=limit, 
                                    device=device, dtype=dtype, requires_grad=True))
        
    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose((1, 2)).transpose((2, 3))
        out = ops.conv(x, self.weight, stride=self.stride, padding=self.padding)
        N, H_out, W_out, C_out = out.shape
        if self.has_bias:
            bias = ops.broadcast_to(ops.reshape(self.bias, (1, C_out)), shape=(W_out, C_out))
            bias = ops.broadcast_to(ops.reshape(bias, (1, W_out, C_out)), shape=(H_out, W_out, C_out))
            out += ops.broadcast_to(ops.reshape(bias, (1, H_out, W_out, C_out)), shape=(N, H_out, W_out, C_out))
        return out.transpose((2, 3)).transpose((1, 2))


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.act = Tanh if nonlinearity == 'tanh' else ReLU
        init_low, init_high = -(1/hidden_size) ** (1/2), (1/hidden_size) ** (1/2)
        self.W_ih = Parameter(
            init.rand(input_size, hidden_size, low=init_low, high=init_high, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(
            init.rand(hidden_size, hidden_size, low=init_low, high=init_high, device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(
                init.rand(hidden_size, low=init_low, high=init_high, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(
                init.rand(hidden_size, low=init_low, high=init_high, device=device, dtype=dtype, requires_grad=True))

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        bs = X.shape[0]
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype, requires_grad=True)
        out = X @ self.W_ih + h @ self.W_hh
        if self.bias:
            out += (ops.broadcast_to(ops.reshape(self.bias_ih, (1, self.hidden_size)), shape=out.shape) + 
                    ops.broadcast_to(ops.reshape(self.bias_hh, (1, self.hidden_size)), shape=out.shape))
        return self.act()(out)


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        self.num_layers = num_layers;
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias=bias,
                                  nonlinearity=nonlinearity, device=device, dtype=dtype)]
        self.rnn_cells += [RNNCell(hidden_size, hidden_size, bias=bias,
                                   nonlinearity=nonlinearity, device=device, dtype=dtype)
                              for _ in range(num_layers-1)]

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        seq_len, bs, _ = X.shape
        X_t = list(ops.split(X, axis=0).tuple())
        h_layers = list(ops.split(h0, axis=0).tuple()) if h0 is not None else [None]*(self.num_layers)
        for t in range(seq_len):
            for l in range(self.num_layers):
                h = self.rnn_cells[l](X_t[t], h_layers[l])
                X_t[t], h_layers[l] = h, h
        return ops.stack(tuple(X_t), axis=0), ops.stack(tuple(h_layers), axis=0)


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        init_low, init_high = -(1/hidden_size) ** (1/2), (1/hidden_size) ** (1/2)
        self.W_ih = Parameter(
            init.rand(input_size, 4*hidden_size, low=init_low, high=init_high, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(
            init.rand(hidden_size, 4*hidden_size, low=init_low, high=init_high, device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(
                init.rand(4*hidden_size, low=init_low, high=init_high, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(
                init.rand(4*hidden_size, low=init_low, high=init_high, device=device, dtype=dtype, requires_grad=True))


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        bs = X.shape[0]
        if h is None or h == (None, None): 
            h0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype, requires_grad=True)
            c0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype, requires_grad=True)
        else:
            h0, c0 = h
        out = X @ self.W_ih + h0 @ self.W_hh
        if self.bias:
            out += (ops.broadcast_to(ops.reshape(self.bias_ih, (1, 4*self.hidden_size)), shape=out.shape) + 
                    ops.broadcast_to(ops.reshape(self.bias_hh, (1, 4*self.hidden_size)), shape=out.shape))
        acts = (Sigmoid(), Sigmoid(), Tanh(), Sigmoid())
        i, f, g, o = tuple(act(elem) 
                          for act, elem in zip(acts, ops.split(out.reshape((bs, 4, self.hidden_size)), axis=1)))
        c_prime = f * c0 + i * g
        h_prime = o * Tanh()(c_prime)
        return h_prime, c_prime



class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        super().__init__()
        self.num_layers = num_layers;
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias=bias, device=device, dtype=dtype)]
        self.lstm_cells += [LSTMCell(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)
                              for _ in range(num_layers-1)]

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            c_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        seq_len, bs, _ = X.shape
        X_t = list(ops.split(X, axis=0).tuple())
        h0, c0 = h if h is not None else (None, None)
        h_layers = list(ops.split(h0, axis=0).tuple()) if h0 is not None else [None]*(self.num_layers)
        c_layers = list(ops.split(c0, axis=0).tuple()) if c0 is not None else [None]*(self.num_layers)
        for t in range(seq_len):
            for l in range(self.num_layers):
                h, c = self.lstm_cells[l](X_t[t], (h_layers[l], c_layers[l]))
                X_t[t], h_layers[l], c_layers[l] = h, h, c
        return ops.stack(tuple(X_t), axis=0), (ops.stack(tuple(h_layers), axis=0), ops.stack(tuple(c_layers), axis=0))


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype="float32"):
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.
        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector
        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
                      init.randn(self.num_embeddings, self.embedding_dim, device=device, dtype=dtype, requires_grad=True))

    def forward(self, X: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors
        Input:
        x of shape (seq_len, bs)
        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        seq_len, bs = X.shape
        one_hot = init.one_hot(n=self.num_embeddings, i=X.reshape((seq_len * bs,)), 
                                device=X.device, dtype=X.dtype)
        # TODO: Try to replace matmul with tensor indexing below                               
        out = one_hot @ self.weight
        return out.reshape((seq_len, bs, self.embedding_dim))
