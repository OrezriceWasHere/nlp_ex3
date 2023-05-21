import torch
from torch.utils.data import ConcatDataset, random_split, DataLoader
import trainer
from data.sequence_dataset import SequenceDataset
from sequence_lstm_net import SequenceLSTMNet
from hyper_parameters import *


def experiment(pos_file, neg_file, parameters):
    print(f'running on {parameters.device}')
    word_to_index = {char: i for i, char in enumerate(parameters.vocab)}

    neg_dataset = SequenceDataset(neg_file, tag=0, word_to_index=word_to_index, parameters=parameters)
    pos_dataset = SequenceDataset(pos_file, tag=1, word_to_index=word_to_index, parameters=parameters)
    dataset = ConcatDataset([neg_dataset, pos_dataset])

    # Split dataset to train and test
    train_size = int(parameters.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, shuffle=True)
    test_dataloader = DataLoader(test_dataset)

    # Initialize SequenceLSTMNet
    model = SequenceLSTMNet(
        vocab_size=parameters.indexes_size,
        embedding_size=parameters.embedding_dim,
        hidden_size=parameters.hidden_size,
        num_layers=parameters.lstm_layers,
        dropout=parameters.dropout,
        output_size=parameters.num_classes
    )
    model = model.to(parameters.device)  # Move model to GPU if available
    loss = torch.nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.lr)

    trainer.train(
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        model=model,
        criterion=loss,
        optimizer=optimizer,
        device=parameters.device,
        epochs=parameters.epochs
    )


def experiment_part_one():
    parameters = P1Parameters
    neg_file = "data/part_one/neg_examples"
    pos_file = "data/part_one/pos_examples"
    experiment(pos_file, neg_file, parameters)


def experiment_part_two_palindrome():
    parameters = P2PalindromeParameters
    neg_file = "data/part_two/palindrome/neg_examples"
    pos_file = "data/part_two/palindrome/pos_examples"
    experiment(pos_file, neg_file, parameters)
    # When sequence length is 10, the model is able to learn the palindrome task with sucess rate 0.94.
    # When seqeunce length is 20, the model is able to learn the palindrome task with sucess rate 0.72.

    # Best results are achieved with 1 LSTM layer and 200 epochs.

def experiment_part_two_power():
    parameters = P3PowerParameters
    neg_file = "data/part_two/power/neg_examples"
    pos_file = "data/part_two/power/pos_examples"
    experiment(pos_file, neg_file, parameters)

    """
    I couldn't reach under any configuration something better than a random guess. 
    The network is not able to learn the power task.
    That makes sense. 
    """

def experiment_part_two_z3():
    parameters = P3Z3Parameters
    neg_file = "data/part_two/z3/neg_examples"
    pos_file = "data/part_two/z3/pos_examples"
    experiment(pos_file, neg_file, parameters)
    """
    The network is able to learn the z3 task with 100% success rate.
    """



if __name__ == "__main__":
    experiment_part_one()
    # experiment_part_two_z3()
