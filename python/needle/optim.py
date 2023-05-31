"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            new_grad = (param.grad.data + param.data * self.weight_decay)
            new_u = self.momentum * self.u.get(param, 0) + (1 - self.momentum) * new_grad
            self.u[param] = new_u
            param.data -= self.lr * new_u

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for param in self.params:
            new_grad = (param.grad.data + param.data * self.weight_decay)
            new_m = self.beta1 * self.m.get(param, 0) + (1 - self.beta1) * new_grad
            new_v = self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * (new_grad ** 2)
            self.m[param], self.v[param] = new_m, new_v
            hat_m = new_m / (1 - self.beta1 ** self.t)
            hat_v = new_v / (1 - self.beta2 ** self.t)
            param.data -= (self.lr * hat_m / (hat_v ** (1/2) + self.eps))

