from torch.utils.data import Dataset, DataLoader
import numpy as np


def _extract_tags(line):
    texts = [text.split("\t") for text in line]
    tokens = [text[0] for text in texts]
    tags = [text[-1] for text in texts]
    return tokens, tags


class Sentense_loader(Dataset):
    """
    training/testing loader
    """

    def __init__(self, file_path):
        f = open(file_path, 'r')
        lines = f.read().split("\n\n")
        lines = [line.split("\n") for line in lines]
        self.data = [_extract_tags(line) for line in lines]
        self.data = np.asarray(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx][0]
        label = self.data[idx][1]
        return {'sentence': sentence, 'label': label}


if __name__ == '__main__':
    loader = Sentense_loader('data/train.txt')
    dataloader = DataLoader(loader, batch_size=4, shuffle=True, num_workers=1)
    for i, data in enumerate(dataloader):
        print(i)
        print(len(data['sentence']))
        print(len(data['label']))
        print(0)
        break
