from torch.utils.data import Dataset, DataLoader
import torch


def _extract_tags(line):
    texts = [text.split("\t") for text in line]
    tokens = [text[0] for text in texts]
    tags = [text[-1] for text in texts]
    return tokens, tags


class Sentense_loader(Dataset):
    """
    training/testing loader
    """

    def __init__(self, file_path, transform=None):
        self.transform = transform
        self.UNKNOWN_WORD = "<UNK>"

        f = open(file_path, 'r')
        lines = f.read().split("\n\n")
        lines = [line.split("\n") for line in lines]
        self.data = [_extract_tags(line) for line in lines]
        self.word_to_ix = {}
        for tagged in self.data:
            sentence = tagged[0]
            for word in sentence:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
        self.word_to_ix[self.UNKNOWN_WORD] = len(self.word_to_ix)

    def prepare_sequence(self, seq):
        idxs = []
        for w in seq:
            if w not in self.word_to_ix:
                w = self.UNKNOWN_WORD
            idxs.append(self.word_to_ix[w])
        return idxs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.prepare_sequence(self.data[idx][0])
        label = self.data[idx][1]
        sample = {'sentence': sentence, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    loader = Sentense_loader('data/train.txt')
    dataloader = DataLoader(loader, batch_size=4, shuffle=True, num_workers=1)
    for i, data in enumerate(dataloader):
        print(i)
        print(len(data['sentence']))
        print(len(data['label']))
        print(0)
        break
