import torch


class SequenceLSTMNet(torch.nn.Module):

    def __init__(self,
                 vocab_size,  # How many distinct characters are there in the vocabulary
                 embedding_size,  # Size of the embedding vector
                 hidden_size,  # Size of the hidden state (= size of the output of the LSTM)
                 num_layers,  # Number of LSTM layers
                 dropout,  # Dropout rate
                 output_size  # Size of the output of the network (= number of classes)
                 ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.lstm = torch.nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        out, hidden = self.lstm(x)
        x = self.linear(out[:, -1, :])
        x = self.sigmoid(x)
        return x
