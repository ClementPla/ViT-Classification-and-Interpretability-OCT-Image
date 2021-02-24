import nntools.dataset as D
import pandas as pd
import torch


class DeitDataset(D.ClassificationDataset):
    def __init__(self, dist_pred_path, *args, **kwargs):
        super(DeitDataset, self).__init__(*args, **kwargs)
        self.csv_table = pd.read_csv(dist_pred_path)
        self.csv_table.set_index('file', inplace=True)

    def __getitem__(self, item):
        parent_items = super(DeitDataset, self).__getitem__(item)
        filename = self.filename(item)

        return (*parent_items, torch.tensor(self.csv_table.loc[filename]['prediction'], dtype=torch.long))


if __name__ == '__main__':
    from nntools.utils import Config
    c = Config('../configs/config.yaml')
    table_path = '/home/clement/Documents/Clement/runs/mlruns/1/f225fafe580e48d491e1eebe122d0635/artifacts' \
                 '/train_predictions_log.csv'
    dataset = DeitDataset(table_path, **c['Dataset'])

    print(dataset[0])