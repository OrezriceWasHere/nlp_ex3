import torch


class Parameters:
    dropout = 0.5
    lstm_layers = 2
    epochs = 50
    lr = 0.001
    train_split = 0.8
    embedding_dim = 10
    hidden_size = 128
    num_classes = 2
    vocab = "123456789abcd"
    indexes_size = len(vocab)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


