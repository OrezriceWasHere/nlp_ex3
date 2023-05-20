
import torch

from hyper_parameters import Parameters


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, sequence_file, tag, word_to_index):
        with open(sequence_file, 'r') as f:
            self.sequences = f.readlines()

        self.sequences = [
            torch.tensor([word_to_index[word] for word in sequence.replace("\n", "")])
            for sequence in self.sequences
        ]
        self.tag = torch.nn.functional.one_hot(torch.tensor(tag), num_classes=Parameters.num_classes).float()
        self.tag = self.tag

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.tag
