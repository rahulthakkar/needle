import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


def ConvBN(in_channels, out_channels, kernel_size, stride, device, dtype):
    return nn.Sequential(
      nn.Conv(in_channels, out_channels, kernel_size, stride, bias=True, device=device, dtype=dtype),
      nn.BatchNorm2d(out_channels, device=device, dtype=dtype),
      nn.ReLU())

def ResidualConvBN(in_channels, out_channels, kernel_size, stride, device, dtype):
    block = nn.Sequential(
      ConvBN(in_channels, out_channels, kernel_size, stride, device=device, dtype=dtype),
      ConvBN(out_channels, out_channels, kernel_size, stride, device=device, dtype=dtype))
    return nn.Residual(block)

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.device= device
        self.dtype = dtype
        self.model = nn.Sequential(
          ConvBN(in_channels=3, out_channels=16, kernel_size=7, stride=4, device=self.device, dtype=self.dtype),
          ConvBN(in_channels=16, out_channels=32, kernel_size=3, stride=2, device=self.device, dtype=self.dtype),
          ResidualConvBN(in_channels=32, out_channels=32, kernel_size=3, stride=1, device=self.device, dtype=self.dtype),
          ConvBN(in_channels=32, out_channels=64, kernel_size=3, stride=2, device=self.device, dtype=self.dtype),
          ConvBN(in_channels=64, out_channels=128, kernel_size=3, stride=2, device=self.device, dtype=self.dtype),
          ResidualConvBN(in_channels=128, out_channels=128, kernel_size=3, stride=1, device=self.device, dtype=self.dtype),
          nn.Flatten(),
          nn.Linear(in_features=128, out_features=128, device=self.device, dtype=self.dtype),
          nn.ReLU(),
          nn.Linear(in_features=128, out_features=10, device=self.device, dtype=self.dtype))

    def forward(self, x):
        return self.model(x)
        

class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        model = nn.RNN if (seq_model == 'rnn') else nn.LSTM
        self.embedding_layer = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        self.seq_model_ = model(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)


    def forward(self, X, h = None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        hidden, last_layer_hidden = self.seq_model_(self.embedding_layer(X), h)
        seq_len, bs, hidden_size = hidden.shape
        return self.linear(hidden.reshape((seq_len * bs, hidden_size))), last_layer_hidden


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)