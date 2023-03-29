from torch.utils.data import DataLoader


class QDataLoader():
    def __init__(self, dataset, **kwargs):
        self._dataset = dataset
        self._kwargs = kwargs

    def dataloader(self, batch_size=1, shuffle=None):
        return DataLoader(
            self._dataset,
            shuffle=shuffle, batch_size=batch_size,
            **self._kwargs
        )
