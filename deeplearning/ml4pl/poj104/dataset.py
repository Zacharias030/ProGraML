# better dataloader
from pathlib import Path

import tqdm
import torch
from torch_geometric.data import InMemoryDataset

class POJ104Dataset(InMemoryDataset):
    def __init__(self, root, split, transform=None, pre_transform=None, train_subset=[0, 100], train_subset_seed=0):
        """
        Args:
            train_subset: [start_percentile, stop_percentile)    default [0,100).
                            sample a random (but fixed) train set of data in slice by percent, with given seed.
            train_subset_seed: seed for the train_subset fixed random permutation.

        """
        self.split = split
        self.train_subset = train_subset
        self.train_subset_seed = train_subset_seed
        super().__init__(root, transform, pre_transform)

        assert split in ['train', 'val', 'test']
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'classifyapp_data.zip' #['ir_val', 'ir_val_programl']

    @property
    def processed_file_names(self):
        base = f'{self.split}_data.pt'

        if tuple(self.train_subset) == (0, 100) or self.split in ['val', 'test']:
            return [base]
        else:
            assert self.split == 'train'
            return [f'{self.split}_data_subset_{self.train_subset[0]}_{self.train_subset[1]}_seed_{self.train_subset_seed}.pt']

    def download(self):
        # download to self.raw_dir
        pass

    def _save_train_subset(self):
        """saves a train_subset of self to file.
        Percentile slice is taken according to self.train_subset
        with a fixed random permutation with self.train_subset_seed.
        """
        import numpy as np
        perm = np.random.RandomState(self.train_subset_seed).permutation(len(self))

        # take slice of perm according to self.train_subset
        start = np.math.floor(len(self) / 100 * self.train_subset[0])
        stop = np.math.floor(len(self) / 100 * self.train_subset[1])
        perm = perm[start:stop]
        print(f'Fixed permutation starts with: {perm[:min(100, len(perm))]}')

        dataset = self.__indexing__(perm)

        data, slices = dataset.data, dataset.slices
        torch.save((data, slices), self.processed_paths[0])
        return

    def process(self):
        # hardcoded
        num_classes = 104

        # check if we need to create the full dataset:
        full_dataset = Path(self.processed_dir) / f'{self.split}_data.pt'
        if full_dataset.is_file():
            assert self.split == 'train', 'here shouldnt be reachable.'
            print(f"Full dataset found. Generating train_subset={self.train_subset} with seed={self.train_subset_seed}")
            # just get the split and save it
            self.data, self.slices = torch.load(full_dataset)
            self._save_train_subset()
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), 'shouldnt be'
        processed_path = str(full_dataset)

        # read data into huge `Data` list.
        data_list = []

        ds_base = Path(self.root)
        print(f'Creating {self.split} dataset at {str(ds_base)}')
        out_base = ds_base / ('ir_' + self.split + '_programl')
        assert out_base.exists(), f"{out_base} doesn't exist!"
        print(f"=== DATASET {out_base}: Collecting .data.p files into dataset")

        folders = [x for x in out_base.glob("*") if x.is_dir() and x.name not in ['_nx', '_tuples']]
        for folder in tqdm.tqdm(folders):
            # skip classes that are larger than what config says to enable debugging with less data
            if int(folder.name) > num_classes:
                continue
            for k, file in enumerate(folder.glob("*.data.p")):
                with open(file, "rb") as f:
                    data = pickle.load(f)
                data_list.append(data)

        print(f" * COMPLETED * === DATASET {out_base}: now pre-filtering...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(f" * COMPLETED * === DATASET {out_base}: Completed filtering, now pre_transforming...")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100) and self.split not in ['val', 'test']:
            self._save_train_subset()



