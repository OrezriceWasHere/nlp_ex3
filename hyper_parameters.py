import torch


class BaseParameters:
    dropout = 0.5
    lstm_layers = 2
    lr = 0.001
    train_split = 0.7
    hidden_size = 128
    num_classes = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class P1Parameters(BaseParameters):
    epochs = 50
    embedding_dim = 10
    vocab = "123456789abcd"
    indexes_size = len(vocab)


class P2PalindromeParameters(BaseParameters):
    epochs = 200
    embedding_dim = 5
    vocab = "abcdefghijklmnopqrstuvwxyz"
    indexes_size = len(vocab)


class P3PrimeParameters(BaseParameters):
    epochs = 300
    embedding_dim = 2
    vocab = "01"
    indexes_size = len(vocab)


class P3Z3Parameters(BaseParameters):
    epochs = 300
    embedding_dim = 2
    vocab = "01"
    indexes_size = len(vocab)