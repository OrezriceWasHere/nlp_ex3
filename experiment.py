import torch
from torch.utils.data import ConcatDataset, random_split, DataLoader
import trainer
from data.sequence_dataset import SequenceDataset
from sequence_lstm_net import SequenceLSTMNet
from hyper_parameters import P1Parameters

if __name__ == "__main__":
    parameters = P1Parameters
    print(f'running on {parameters.device}')
    neg_file = "data/part_one/neg_examples"
    pos_file = "data/part_one/pos_examples"
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
