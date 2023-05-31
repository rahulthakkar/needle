"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad,


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * self.scalar,


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        a, = node.inputs
        return out_grad * self.scalar * (a ** (self.scalar-1)),

def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / (rhs **2)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar,


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int): 
            axes = (axes,)
        self.axes = axes

    def compute(self, a):
        dims = len(a.shape)
        axes = (dims-1, dims-2) if self.axes is None else self.axes
        return array_api.swapaxes(a, *axes)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes),


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a.compact(), self.shape)

    def gradient(self, out_grad, node):
        a, = node.inputs
        return reshape(out_grad, a.shape),


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        a, = node.inputs
        if out_grad.shape == a.shape:
            return out_grad,
        extra_dims = (1,) * (len(out_grad.shape) - len(a.shape))
        axes = tuple(idx for idx, (in_dim, out_dim)
                     in enumerate(zip(out_grad.shape, extra_dims+a.shape))
                     if in_dim!=out_dim)
        return reshape(summation(out_grad, axes), a.shape),

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int): 
            axes = (axes,)
        self.axes = axes

    def compute(self, a):
        if self.axes == tuple():
            return a
        return a.sum(self.axes)

    def gradient(self, out_grad, node):
        a, = node.inputs
        return broadcast_squeezed(out_grad, a.shape, self.axes),


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a, grad_b = out_grad @ transpose(b), transpose(a) @ out_grad
        sum_a_axes = tuple(i for i in range(len(b.shape) - len(a.shape)))
        sum_b_axes = tuple(i for i in range(len(a.shape) - len(b.shape)))
        return summation(grad_a, axes=sum_a_axes), summation(grad_b, axes=sum_b_axes)

def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad,


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        a, = node.inputs
        return out_grad / a,

def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        a, = node.inputs
        out = node.realize_cached_data()
        return out * out_grad,


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        a, = node.inputs
        grad_flag = array_api.array(a.realize_cached_data() > 0, dtype=a.dtype, device=a.device)
        return out_grad * Tensor.make_const(grad_flag),


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int): 
            axes = (axes,)
        self.axes = axes

    def compute(self, Z):
        self.Z_max = Z.max(axis=self.axes, keepdims=True)
        Z_diff = Z - self.Z_max.broadcast_to(Z.shape)
        log_sum_exp = Z_diff.exp().sum(axis=self.axes, keepdims=True).log() + self.Z_max
        squeezed_shape = tuple(dim for i, dim in enumerate(Z.shape) 
                                if (self.axes is None or i not in self.axes))
        return log_sum_exp.reshape(squeezed_shape)
        # return log_sum_exp

    def gradient(self, out_grad, node):
        Z, = node.inputs
        Z = Z - Tensor(self.Z_max.broadcast_to(Z.shape), device=Z.device, dtype=Z.dtype)
        out_broadcasted = broadcast_squeezed(out_grad, Z.shape, self.axes)
        return (exp(Z) / broadcast_squeezed(summation(exp(Z), axes=self.axes), Z.shape, self.axes)) * out_broadcasted,


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


def broadcast_squeezed(tensor, broadcasted_shape, squeezed_axes):
    broadcastable_shape = tuple(1 if (squeezed_axes is None or i in squeezed_axes)
                                  else s
                                for i, s in enumerate(broadcasted_shape))
    return broadcast_to(reshape(tensor, shape=broadcastable_shape), shape=broadcasted_shape)

class Tanh(TensorOp):
    def compute(self, a):
        exp_2a = array_api.exp(2*a)
        return (exp_2a - 1) / (exp_2a + 1)
      
    def gradient(self, out_grad, node):
        out = node.realize_cached_data()
        return out_grad * (1. - out**2),


def tanh(a):
    return Tanh()(a)


class Sigmoid(TensorOp):
    def compute(self, a):
        exp_a = array_api.exp(a)
        return exp_a / (1 + array_api.exp(a))
      
    def gradient(self, out_grad, node):
        out = node.realize_cached_data()
        return out_grad * out * (1. - out),


def sigmoid(a):
    return Sigmoid()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ## TODO Check if all args have the same shape (extend to allow different shapes?) 
        ## and are on the same device
        new_dim = len(args)
        arr_shape = args[0].shape
        stacked_shape = arr_shape[:self.axis] + (new_dim,) + arr_shape[self.axis:]
        out = array_api.empty(stacked_shape, dtype=args[0].dtype, device=args[0].device) 
        slices_before = tuple(slice(dim) for dim in arr_shape[:self.axis])
        slices_after = tuple(slice(dim) for dim in arr_shape[self.axis:])
        for i in range(new_dim):
            curr_slices = slices_before  + (slice(i, i+1),) + slices_after
            out[curr_slices] = args[i] 
        return out

    def gradient(self, out_grad, node):
        return split(out_grad, self.axis)

def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        splitted_shape = A.shape[:self.axis] + A.shape[self.axis+1:]
        slices_before = tuple(slice(dim) for dim in A.shape[:self.axis])
        slices_after = tuple(slice(dim) for dim in A.shape[self.axis+1:])
        out = tuple(A[slices_before + (slice(i, i+1),) + slices_after].compact().reshape(splitted_shape)
                      for i in range(A.shape[self.axis]))
        return out

    def gradient(self, out_grads, node):    
        return stack(out_grads, self.axis),

def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.flip(a, self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes),


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        new_shape = tuple((dim * (self.dilation+1)) if i in self.axes else dim
                           for i, dim in enumerate(a.shape))
        out = array_api.zeros(new_shape, device=a.device) 
        index = tuple(slice(None, None, self.dilation+1) if i in self.axes else slice(None)
                      for i in range(len(a.shape)))
        out[index] = a
        return out
        
    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation),

def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        new_shape = tuple((dim // (self.dilation+1)) if i in self.axes else dim
                           for i, dim in enumerate(a.shape))
        out = array_api.zeros(new_shape, device=a.device) 
        index = tuple(slice(None, None, self.dilation+1) if i in self.axes else slice(None)
                      for i in range(len(a.shape)))
        out = a[index]
        return out

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation),


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, x, weight):
        if self.padding > 0:
            padding_tup = (self.padding, self.padding)
            axes = ((0,0), padding_tup, padding_tup, (0,0))
            x = x.pad(axes)
        N, H, W, C_in = x.shape
        K, _, _, C_out = weight.shape
        Ns, Hs, Ws, C_ins = x.strides
        s = 1 if self.stride is None else self.stride
        H_out, W_out = ((H - K) // s) + 1, ((W - K) // s) + 1
        inner_dim = K * K * C_in
        x = x.as_strided(shape=(N, H_out, W_out, K, K, C_in), 
                        strides=(Ns, Hs*s, Ws*s, Hs, Ws, C_ins)).compact()
        out = x.reshape((N * H_out * W_out, inner_dim)) @ weight.compact().reshape((inner_dim, C_out))
        return out.reshape((N, H_out, W_out, C_out))

    def gradient(self, out_grad, node):
        x, weight = node.inputs
        N, H, W, C_in = x.shape
        K, _, _, C_out = weight.shape
        p = self.padding
        weight = flip(weight, axes=(0, 1)).transpose((2, 3)) # K, K, C_out, C_in
        if self.stride > 1:
            out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride-1) # N, H-K+1, W-K+1, C_out
        grad_x = conv(out_grad, weight, padding=(K - p - 1))
        x = x.transpose(axes=(0, 3)) # C_in, H, W, N
        out_grad = out_grad.transpose(axes=(0, 1)).transpose(axes=(1, 2)) # H-K+1, W-K+1, N, C_out
        grad_weight = conv(x, out_grad, padding=p) # C_in, K, K, C_out
        grad_weight = grad_weight.transpose(axes=(0, 1)).transpose(axes=(1, 2))
        return grad_x, grad_weight
        

def conv(x, weight, stride=1, padding=1):
    return Conv(stride, padding)(x, weight)



