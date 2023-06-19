from jittor.dataset import Dataset


class RepeatDataset():
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):

        self.dataset_ = dataset
        self.times = times
        # self.CLASSES = dataset.CLASSES

        self._ori_len = len(self.dataset_)


    def __getitem__(self, idx):
        return self.dataset_[idx % self._ori_len]

    def get_cat_ids(self, idx):
        return self.dataset_.get_cat_ids(idx % self._ori_len)

    def __len__(self):
        return self.times * self._ori_len

    def __iter__(self):
        for _ in range(self.times):
            for idxs in range(self._ori_len):
                yield self.dataset_[idxs]


