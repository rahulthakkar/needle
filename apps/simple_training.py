import sys
sys.path.append('../python')
import math
import numpy as np
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
from time import time
from tqdm import trange, tqdm

device = ndl.cpu()

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    losses = []
    correct_pred_count, total_count = 0, 0
    device = model.device
    if opt:
        model.train()
    else:
        model.eval()
    for X_batch, y_batch in tqdm(dataloader, total=dataloader.num_batches, position=0, leave=True):
        X_batch, y_batch = ndl.Tensor(X_batch, device=device), ndl.Tensor(y_batch, device=device),
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        losses.append(loss.detach().numpy())
        correct_pred_count += np.sum(logits.detach().numpy().argmax(axis=-1) == y_batch.numpy())
        total_count += X_batch.shape[0]
        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()
    return (correct_pred_count / total_count), np.mean(losses)


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
                  lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss(), print_stats=False):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(n_epochs):
        print(f"Epoch {i+1}: ")
        start = time()
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn=loss_fn, opt=opt)
        end = time()
        if print_stats:
          print(f"Average Epoch Loss: {avg_loss:.2f} || ", 
                f"Average Epoch Accuracy: {avg_acc:.2f} || ",
                f"Time: {end-start:.0f}s")
    return avg_acc, avg_loss


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss()):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn=loss_fn, opt=None)
    return avg_acc, avg_loss


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
                      clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.
    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)
    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    nbatch, _ = data.shape
    # losses = []
    correct_pred_count, total_loss, total_count = 0, 0, 0
    hidden = None
    if opt:
        model.train()
    else:
        model.eval()
    for i in trange(0, nbatch, seq_len):
        X_batch, y_batch = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        logits, hidden = model(X_batch, hidden)
        loss = loss_fn(logits, y_batch)
        # losses.append(loss.detach().numpy())
        correct_pred_count += np.sum(logits.detach().numpy().argmax(axis=-1) == y_batch.numpy())
        total_loss += loss.detach().numpy() *  y_batch.shape[0]
        total_count += y_batch.shape[0]
        if opt:
            if isinstance(hidden, tuple): 
                hidden = tuple(elem.detach() for elem in hidden)
            else:
                hidden = hidden.detach()
            opt.reset_grad()
            loss.backward()
            if clip is not None and hasattr(opt, 'clip_grad_norm'):
                opt.clip_grad_norm(clip)
            opt.step()
    return (correct_pred_count / total_count), total_loss / total_count #np.mean(losses)

# TODO Why is default LR so high?
def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
              lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss(), clip=None,
              device=None, dtype="float32", print_stats=False):
    """
    Performs {n_epochs} epochs of training.
    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)
    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(n_epochs):
        print(f"Epoch {i+1}: ")
        start = time()
        avg_acc, avg_loss =  epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn, opt=opt,
                                               clip=clip, device=device, dtype=dtype)
        end = time()
        if print_stats:
          print(f"Average Epoch Loss: {avg_loss:.2f} || ", 
                f"Average Epoch Accuracy: {avg_acc:.2f} || ",
                f"Time: {end-start:.0f}s")
    return avg_acc, avg_loss


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss(),
                 device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.
    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class
    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    avg_acc, avg_loss =  epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn,
                                           device=device, dtype=dtype)
    return avg_acc, avg_loss


if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cpu()
    #dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    #dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    #model = ResNet9(device=device, dtype="float32")
    #train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
