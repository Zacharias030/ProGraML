# better dataloader
from pathlib import Path
import pickle
import math

import tqdm
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import StratifiedKFold, KFold
import torch
from torch_geometric.data import InMemoryDataset, Data


# make this file executable from anywhere

import sys, os
full_path = os.path.realpath(__file__)
#print(full_path)
REPO_ROOT = full_path.rsplit('ProGraML', maxsplit=1)[0] + 'ProGraML'
#print(REPO_ROOT)
#insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, REPO_ROOT)
REPO_ROOT = Path(REPO_ROOT)


def load(file):
    with open(file, 'rb') as f:
        try:
            data = pickle.load(f)
        except EOFError as e:
            print(f"Failing on {str(file)}")
            raise e
    return data


def nx2data(nx_graph, class_label=None, ignore_profile_info=True):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G               (networkx.Graph or networkx.DiGraph): A networkx graph.
        class_label     optional 'y' label. Should be int [0, ..., num_classes - 1]
    """

    # collect edge_index
    edge_index = torch.tensor(list(nx_graph.edges())).t().contiguous()

    # collect edge_attr
    positions = []
    flows = []

    for i, (_, _, edge_data) in enumerate(nx_graph.edges(data=True)):
        flows.append(edge_data['flow'])
        # TODO(remove): this hack fixes a bug where call edges have position info!
        # depends on merge of this fix https://github.com/ChrisCummins/phd/pull/106
        if edge_data['flow'] == 2:
            positions.append(0)
        else:
            positions.append(edge_data['position'])

    positions = torch.tensor(positions)
    flows = torch.tensor(flows)

    edge_attr = torch.cat([flows, positions]).view(2, -1).t().contiguous()

    # collect x
    types = []
    xs = []

    for i, node_data in nx_graph.nodes(data=True):
        types.append(node_data['type'])
        xs.append(node_data['x'][0])

    xs = torch.tensor(xs)
    types = torch.tensor(types)

    x = torch.cat([xs, types]).view(2, -1).t().contiguous()

    assert edge_attr.size()[0] == edge_index.size()[1], f'edge_attr={edge_attr.size()} size mismatch with edge_index={edge_index.size()}'

    data_dict = {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
    }
    
    # maybe collect these data too
    if class_label is not None:
        y = torch.tensor(int(class_label)).view(1)  # <1>
        data_dict['y'] = y
    
    # branch prediction / profile info specific
    if not ignore_profile_info:
        profile_info = []
        for i, node_data in nx_graph.nodes(data=True):
            # default to -1, -1, -1 if not all profile info is given.
            if not (node_data.get("llvm_profile_true_weight") is not None and \
                    node_data.get("llvm_profile_false_weight") is not None and \
                    node_data.get("llvm_profile_total_weight") is not None):
                mask = 0
                true_weight = -1
                false_weight = -1
                total_weight = -1
            else:
                mask = 1
                true_weight = node_data["llvm_profile_true_weight"]
                false_weight = node_data["llvm_profile_false_weight"]
                total_weight = node_data["llvm_profile_total_weight"]

            profile_info.append([mask, true_weight, false_weight, total_weight])
        
        data_dict['profile_info'] = torch.tensor(profile_info)
    
    
    # make Data
    data = Data(**data_dict)

    return data


class BranchPredictionDataset(InMemoryDataset):
    def __init__(self, root='deeplearning/ml4pl/poj104/branch_prediction_data',
                 split='train',
                 transform=None, pre_transform=None,
                 train_subset=[0, 100],
                 train_subset_seed=0):
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

        assert split in ['train'], "The BranchPrediction dataset only has a 'train' split. use train_subset=[0,x] and [x, 100] for training and testing."
        self.data, self.slices = torch.load(self.processed_paths[0])
        pass

    @property
    def raw_file_names(self):
        """A list of files that need to be found in the raw_dir in order to skip the download"""
        return []  # not implemented here

    @property
    def processed_file_names(self):
        """A list of files in the processed_dir which needs to be found in order to skip the processing."""
        base = f'{self.split}_data.pt'

        if tuple(self.train_subset) == (0, 100) or self.split in ['val', 'test']:
            return [base]
        else:
            assert self.split == 'train'
            return [f'{self.split}_data_subset_{self.train_subset[0]}_{self.train_subset[1]}_seed_{self.train_subset_seed}.pt']

    def download(self):
        """Download raw data to `self.raw_dir`"""
        pass  # not implemented

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
        print(f'Fixed permutation starts with: {perm[:min(30, len(perm))]}')

        dataset = self.__indexing__(perm)

        data, slices = dataset.data, dataset.slices
        torch.save((data, slices), self.processed_paths[0])
        return

    def return_cross_validation_splits(self, split):
        assert self.train_subset == [0, 100], "Do cross-validation on the whole dataset!"
        #num_samples = len(self)
        #perm = np.random.RandomState(self.train_subset_seed).permutation(len(self))

         # 10-fold cross-validation
        n_splits = 10
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        (train_index, test_index) = list(kf.split(range(len(self))))[split]
        train_data = self.__indexing__(train_index)
        test_data = self.__indexing__(test_index)
        return train_data, test_data

    def filter_max_num_nodes(self, max_num_nodes):
        idx = []
        for i, d in enumerate(self):
            if d.num_nodes <= max_num_nodes:
                idx.append(i)
        dataset = self.__indexing__(idx)
        print(f"Filtering out graphs larger than {max_num_nodes} yields a dataset with {len(dataset)}/{len(self)} samples remaining.")
        return dataset

    def process(self):
        """Processes raw data and saves it into the `processed_dir`.
        New implementation:
            Here specifically it will collect all '*.ll.pickle' files recursively from subdirectories of `root`
            and process the loaded nx graphs to Data.
        Old implementation:
            Instead of looking for .ll.pickle (nx graphs), we directly look for '*.data.p' files.
        """
        # check if we need to create the full dataset:
        full_dataset = Path(self.processed_dir) / f'{self.split}_data.pt'
        if full_dataset.is_file():
            assert self.split == 'train', 'here shouldnt be reachable.'
            print(f"Full dataset found. Generating train_subset={self.train_subset} with seed={self.train_subset_seed}")
            # just get the split and save it
            self.data, self.slices = torch.load(full_dataset)
            self._save_train_subset()
            print(f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk.")
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), 'shouldnt be'
        processed_path = str(full_dataset)

        # read data into huge `Data` list.
        data_list = []

        ds_base = Path(self.root)
        print(f'Creating {self.split} dataset at {str(ds_base)}')
        # TODO change this line to go to the new format
        #out_base = ds_base / ('ir_' + self.split + '_programl')
        #assert out_base.exists(), f"{out_base} doesn't exist!"
        # TODO collect .ll.pickle instead and call nx2data on the fly!
        print(f"=== DATASET {str(ds_base)}: Collecting .data.p files into dataset")

        #files = list(ds_base.rglob('*.data.p'))
        #files = list(ds_base.rglob('*.ll.pickle'))
        files = list(ds_base.rglob('*.ll.p'))
        
        for file in tqdm.tqdm(files):
            if not file.is_file():
                continue
            try:
                nx_graph = load(file)
            except EOFError:
                print(f"Failing to unpickle bc. EOFError on {file}! Skipping ...")
                continue
            try:
                data = nx2data(nx_graph, ignore_profile_info=False)
                data_list.append(data)
            except IndexError:
                print(f"Failing nx2data bc IndexError (prob. empty graph) on {file}! Skipping ...")
                continue

        print(f" * COMPLETED * === DATASET {ds_base}: now pre-filtering...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(f" * COMPLETED * === DATASET {ds_base}: Completed filtering, now pre_transforming...")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        print(f" * COMPLETED * === DATASET {ds_base}: saving to disk...")
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100) and self.split not in ['val', 'test']:
            self._save_train_subset()



class NCCDataset(InMemoryDataset):
    def __init__(self, root=REPO_ROOT / 'deeplearning/ml4pl/poj104/ncc_data',
                 split='train',
                 transform=None, pre_transform=None,
                 train_subset=[0, 100],
                 train_subset_seed=0):
        """
        NCC dataset

        Args:
            train_subset: [start_percentile, stop_percentile)    default [0,100).
                            sample a random (but fixed) train set of data in slice by percent, with given seed.
            train_subset_seed: seed for the train_subset fixed random permutation.

        """
        self.split = split
        self.train_subset = train_subset
        self.train_subset_seed = train_subset_seed
        super().__init__(root, transform, pre_transform)

        assert split in ['train'], "The NCC dataset only has a 'train' split. use train_subset=[0,x] and [x, 100] for training and testing."
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """A list of files that need to be found in the raw_dir in order to skip the download"""
        return []  # not implemented here

    @property
    def processed_file_names(self):
        """A list of files in the processed_dir which needs to be found in order to skip the processing."""
        base = f'{self.split}_data.pt'

        if tuple(self.train_subset) == (0, 100) or self.split in ['val', 'test']:
            return [base]
        else:
            assert self.split == 'train'
            return [f'{self.split}_data_subset_{self.train_subset[0]}_{self.train_subset[1]}_seed_{self.train_subset_seed}.pt']

    def download(self):
        """Download raw data to `self.raw_dir`"""
        pass  # not implemented

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
        print(f'Fixed permutation starts with: {perm[:min(30, len(perm))]}')

        dataset = self.__indexing__(perm)

        data, slices = dataset.data, dataset.slices
        torch.save((data, slices), self.processed_paths[0])
        return

    def filter_max_num_nodes(self, max_num_nodes):
        idx = []
        for i, d in enumerate(self):
            if d.num_nodes <= max_num_nodes:
                idx.append(i)
        dataset = self.__indexing__(idx)
        print(f"Filtering out graphs larger than {max_num_nodes} yields a dataset with {len(dataset)}/{len(self)} samples remaining.")
        return dataset

    def process(self):
        """Processes raw data and saves it into the `processed_dir`.
        New implementation:
            Here specifically it will collect all '*.ll.pickle' files recursively from subdirectories of `root`
            and process the loaded nx graphs to Data.
        Old implementation:
            Instead of looking for .ll.pickle (nx graphs), we directly look for '*.data.p' files.
        """
        # check if we need to create the full dataset:
        full_dataset = Path(self.processed_dir) / f'{self.split}_data.pt'
        if full_dataset.is_file():
            assert self.split == 'train', 'here shouldnt be reachable.'
            print(f"Full dataset found. Generating train_subset={self.train_subset} with seed={self.train_subset_seed}")
            # just get the split and save it
            self.data, self.slices = torch.load(full_dataset)
            self._save_train_subset()
            print(f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk.")
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), 'shouldnt be'
        processed_path = str(full_dataset)

        # read data into huge `Data` list.
        data_list = []

        ds_base = Path(self.root)
        print(f'Creating {self.split} dataset at {str(ds_base)}')
        # TODO change this line to go to the new format
        #out_base = ds_base / ('ir_' + self.split + '_programl')
        #assert out_base.exists(), f"{out_base} doesn't exist!"
        # TODO collect .ll.pickle instead and call nx2data on the fly!
        print(f"=== DATASET {str(ds_base)}: Collecting .data.p files into dataset")

        #files = list(ds_base.rglob('*.data.p'))
        #files = list(ds_base.rglob('*.ll.pickle'))
        files = list(ds_base.rglob('*.ll.p'))
        
        for file in tqdm.tqdm(files):
            if not file.is_file():
                continue
            try:
                nx_graph = load(file)
            except EOFError:
                print(f"Failing to unpickle bc. EOFError on {file}! Skipping ...")
                continue
            try:
                data = nx2data(nx_graph)
                data_list.append(data)
            except IndexError:
                print(f"Failing nx2data bc IndexError (prob. empty graph) on {file}! Skipping ...")
                continue

        print(f" * COMPLETED * === DATASET {ds_base}: now pre-filtering...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(f" * COMPLETED * === DATASET {ds_base}: Completed filtering, now pre_transforming...")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        print(f" * COMPLETED * === DATASET {ds_base}: saving to disk...")
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100) and self.split not in ['val', 'test']:
            self._save_train_subset()




class LegacyNCCDataset(InMemoryDataset):
    def __init__(self, root='deeplearning/ml4pl/poj104/unsupervised_ncc_data',
                 split='train',
                 transform=None, pre_transform=None,
                 train_subset=[0, 100],
                 train_subset_seed=0):
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

        assert split in ['train'], "The NCC dataset only has a 'train' split. use train_subset=[0,x] and [x, 100] for training and testing."
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """A list of files that need to be found in the raw_dir in order to skip the download"""
        return []  # not implemented here

    @property
    def processed_file_names(self):
        """A list of files in the processed_dir which needs to be found in order to skip the processing."""
        base = f'{self.split}_data.pt'

        if tuple(self.train_subset) == (0, 100) or self.split in ['val', 'test']:
            return [base]
        else:
            assert self.split == 'train'
            return [f'{self.split}_data_subset_{self.train_subset[0]}_{self.train_subset[1]}_seed_{self.train_subset_seed}.pt']

    def download(self):
        """Download raw data to `self.raw_dir`"""
        pass  # not implemented

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
        print(f'Fixed permutation starts with: {perm[:min(30, len(perm))]}')

        dataset = self.__indexing__(perm)

        data, slices = dataset.data, dataset.slices
        torch.save((data, slices), self.processed_paths[0])
        return

    def filter_max_num_nodes(self, max_num_nodes):
        idx = []
        for i, d in enumerate(self):
            if d.num_nodes <= max_num_nodes:
                idx.append(i)
        dataset = self.__indexing__(idx)
        print(f"Filtering out graphs larger than {max_num_nodes} yields a dataset with {len(dataset)}/{len(self)} samples remaining.")
        return dataset

    def process(self):
        """Processes raw data and saves it into the `processed_dir`.
        New implementation:
            Here specifically it will collect all '*.ll.pickle' files recursively from subdirectories of `root`
            and process the loaded nx graphs to Data.
        Old implementation:
            Instead of looking for .ll.pickle (nx graphs), we directly look for '*.data.p' files.
        """
        # check if we need to create the full dataset:
        full_dataset = Path(self.processed_dir) / f'{self.split}_data.pt'
        if full_dataset.is_file():
            assert self.split == 'train', 'here shouldnt be reachable.'
            print(f"Full dataset found. Generating train_subset={self.train_subset} with seed={self.train_subset_seed}")
            # just get the split and save it
            self.data, self.slices = torch.load(full_dataset)
            self._save_train_subset()
            print(f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk.")
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), 'shouldnt be'
        processed_path = str(full_dataset)

        # read data into huge `Data` list.
        data_list = []

        ds_base = Path(self.root)
        print(f'Creating {self.split} dataset at {str(ds_base)}')
        # TODO change this line to go to the new format
        #out_base = ds_base / ('ir_' + self.split + '_programl')
        #assert out_base.exists(), f"{out_base} doesn't exist!"
        # TODO collect .ll.pickle instead and call nx2data on the fly!
        print(f"=== DATASET {str(ds_base)}: Collecting .data.p files into dataset")

        files = list(ds_base.rglob('*.data.p'))
        for file in tqdm.tqdm(files):
            if not file.is_file():
                continue
            data = load(file)
            data_list.append(data)

        print(f" * COMPLETED * === DATASET {ds_base}: now pre-filtering...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(f" * COMPLETED * === DATASET {ds_base}: Completed filtering, now pre_transforming...")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        print(f" * COMPLETED * === DATASET {ds_base}: saving to disk...")
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100) and self.split not in ['val', 'test']:
            self._save_train_subset()



class ThreadcoarseningDataset(InMemoryDataset):
    def __init__(self, root='deeplearning/ml4pl/poj104/threadcoarsening_data',
                 split='fail_fast',
                 transform=None, pre_transform=None,
                 train_subset=[0, 100], train_subset_seed=0):
        """
        Args:
            train_subset: [start_percentile, stop_percentile)    default [0,100).
                            sample a random (but fixed) train set of data in slice by percent, with given seed.
            train_subset_seed: seed for the train_subset fixed random permutation.
            split: 'amd' or 'nvidia'

        """
        assert split in ["Cypress", "Tahiti", "Fermi", "Kepler"], f"Split is {split}, but has to be 'Cypress', 'Tahiti', 'Fermi', or  'Kepler'"
        self.split = split
        self.train_subset = train_subset
        self.train_subset_seed = train_subset_seed
        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'threadcoarsening_data.zip'

    @property
    def processed_file_names(self):
        base = f'{self.split}_data.pt'

        if tuple(self.train_subset) == (0, 100):
            return [base]
        else:
            return [f'{self.split}_data_subset_{self.train_subset[0]}_{self.train_subset[1]}_seed_{self.train_subset_seed}.pt']

    def download(self):
        # download to self.raw_dir
        pass

    def return_cross_validation_splits(self, split):
        assert self.train_subset == [0, 100], "Do cross-validation on the whole dataset!"
        assert split <= 16 and split >= 0, f"This dataset shall be 17-fold (leave one out) cross-validated, but split={split}."
        # leave one out
        n_splits = 17
        train_idx = list(range(n_splits))
        train_idx.remove(split)
        train_data = self.__indexing__(train_idx)
        test_data = self.__indexing__([split])
        return train_data, test_data


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

    def platform2str(self, platform):
        if platform == "Fermi":
            return "NVIDIA GTX 480"
        elif platform == "Kepler":
            return "NVIDIA Tesla K20c"
        elif platform == "Cypress":
            return "AMD Radeon HD 5900"
        elif platform == "Tahiti":
            return "AMD Tahiti 7970"
        else:
            raise LookupError

    def _get_all_runtimes(self, platform, df, oracles):
        all_runtimes = {}
        for kernel in oracles['kernel']:
            kernel_r = []
            for cf in [1, 2, 4, 8, 16, 32]:
                row = df[(df['kernel'] == kernel) & (df['cf'] == cf)]
                if len(row) == 1:
                    value = float(row[f'runtime_{platform}'].values)
                    if math.isnan(value):
                        print(f"WARNING: Dataset contain NaN value (missing entry in runtimes most likely). kernel={kernel}, cf={cf}, value={row}.Replacing by infinity!.")
                        value = float('inf')
                    kernel_r.append(value)
                elif len(row) == 0:
                    print(f' kernel={kernel:>20} is missing cf={cf}. Ad-hoc inserting result from last existing coarsening factor.')
                    kernel_r.append(kernel_r[-1])
                else:
                    raise
            all_runtimes[kernel] = kernel_r
        return all_runtimes

    def process(self):
        # check if we need to create the full dataset:
        full_dataset = Path(self.processed_dir) / f'{self.split}_data.pt'
        if full_dataset.is_file():
            print(f"Full dataset {full_dataset.name} found. Generating train_subset={self.train_subset} with seed={self.train_subset_seed}")
            # just get the split and save it
            self.data, self.slices = torch.load(full_dataset)
            self._save_train_subset()
            print(f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk.")
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), 'shouldnt be'
        processed_path = str(full_dataset)

        root = Path(self.root)
        # Load runtime data
        oracle_file = root / "pact-2014-oracles.csv"
        oracles = pd.read_csv(oracle_file)

        runtimes_file = root / "pact-2014-runtimes.csv"
        df = pd.read_csv(runtimes_file)

        print('\tReading data from', oracle_file, '\n\tand', runtimes_file)

        # get all runtime info per kernel
        runtimes = self._get_all_runtimes(self.split, df=df, oracles=oracles)

        # get oracle labels
        cfs = [1, 2, 4, 8, 16, 32]
        y = np.array([cfs.index(int(x)) for x in oracles["cf_" + self.split]], dtype=np.int64)

        # sanity check oracles against min runtimes
        for i, (k, v) in enumerate(runtimes.items()):
            assert int(y[i]) == np.argmin(v), f"{i}: {k} {v}, argmin(v): {np.argmin(v)} vs. oracles data {int(y[i])}."

        # Add attributes to graphs
        data_list = []

        kernels = oracles["kernel"].values  # list of strings of kernel names

        for kernel in kernels:
            #legacy
            #file = root / 'kernels_ir_programl' / (kernel + '.data.p')
            file = root / 'kernels_ir' / (kernel + '.ll.p')
            assert file.exists(), f'input file not found: {file}'
            #with open(file, 'rb') as f:
            #    data = pickle.load(f)
            g = load(file)
            data = nx2data(g)
            # add attributes
            data['y'] = torch.tensor([np.argmin(runtimes[kernel])], dtype=torch.long)
            data['runtimes'] = torch.tensor([runtimes[kernel]])
            data_list.append(data)

        ##################################

        print(f" * COMPLETED * === DATASET Threadcoarsening-{self.split}: now pre-filtering...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(f" * COMPLETED * === DATASET Threadcoarsening-{self.split}: Completed filtering, now pre_transforming...")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100) and self.split not in ['val', 'test']:
            self._save_train_subset()



class DevmapDataset(InMemoryDataset):
    def __init__(self, root='deeplearning/ml4pl/poj104/devmap_data',
                 split='fail', transform=None, pre_transform=None,
                 train_subset=[0, 100], train_subset_seed=0):
        """
        Args:
            train_subset: [start_percentile, stop_percentile)    default [0,100).
                            sample a random (but fixed) train set of data in slice by percent, with given seed.
            train_subset_seed: seed for the train_subset fixed random permutation.
            split: 'amd' or 'nvidia'

        """
        assert split in ['amd', 'nvidia'], f"Split is {split}, but has to be 'amd' or 'nvidia'"
        self.split = split
        self.train_subset = train_subset
        self.train_subset_seed = train_subset_seed
        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'devmap_data.zip'

    @property
    def processed_file_names(self):
        base = f'{self.split}_data.pt'

        if tuple(self.train_subset) == (0, 100):
            return [base]
        else:
            return [f'{self.split}_data_subset_{self.train_subset[0]}_{self.train_subset[1]}_seed_{self.train_subset_seed}.pt']

    def download(self):
        # download to self.raw_dir
        pass

    def return_cross_validation_splits(self, split):
        assert self.train_subset == [0, 100], "Do cross-validation on the whole dataset!"
        #num_samples = len(self)
        #perm = np.random.RandomState(self.train_subset_seed).permutation(len(self))

         # 10-fold cross-validation
        n_splits = 10
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        (train_index, test_index) = list(kf.split(self.data.y, self.data.y))[split]
        train_data = self.__indexing__(train_index)
        test_data = self.__indexing__(test_index)
        return train_data, test_data

    def _save_train_subset(self):
        """saves a train_subset of self to file.
        Percentile slice is taken according to self.train_subset
        with a fixed random permutation with self.train_subset_seed.
        """
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
        # check if we need to create the full dataset:
        full_dataset = Path(self.processed_dir) / f'{self.split}_data.pt'
        if full_dataset.is_file():
            print(f"Full dataset {full_dataset.name} found. Generating train_subset={self.train_subset} with seed={self.train_subset_seed}")
            # just get the split and save it
            self.data, self.slices = torch.load(full_dataset)
            self._save_train_subset()
            print(f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk.")
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), 'shouldnt be'
        processed_path = str(full_dataset)


        root = Path(self.root)

        # Load runtime data
        data_file = root / f"cgo17-{self.split}.csv"
        print('\n--- Read data from', data_file)
        df = pd.read_csv(data_file)
        # Get list of source file names and attributes
        input_files = df["benchmark"].values   # list of strings of benchmark names
        dataset = df["dataset"].values         # list of strings of dataset descriptions
        aux_transfer_size = df["transfer"].values
        aux_wg_size = df["wgsize"].values
        oracle = df['oracle'].values
        runtime_cpu = df['runtime_cpu'].values
        runtime_gpu = df['runtime_gpu'].values

        num_files = len(input_files)
        print('\n--- Preparing to read', num_files, 'input files')

        # read data into huge `Data` list.

        data_list = []
        for i in tqdm.tqdm(range(num_files)):
            filename = input_files[i]
            dat = dataset[i]
            if filename[:3] == "npb":
                # concatenate data set size
                filename += '_' + str(dat)

            # Updated from legacy
            #file = Path(self.root) / 'kernels_ir_programl' / (filename + '.data.p')
            file = Path(self.root) / 'kernels_ir' / (filename + '.ll.p')
            
            if file.exists():
                # legacy
                # load preprocessed data (without labels etc.)
                #with open(file, 'rb') as f:
                #    data = pickle.load(f)
                # new
                g = load(file)
                data = nx2data(g)
            else:
                assert False, f'input file not found: {str(file)}. Did you run data preprocessing?'

            # add data
            data['y'] = torch.tensor([1]) if oracle[i] == 'GPU' else torch.tensor([0]) # CPU
            data['aux_in'] = torch.tensor([[aux_transfer_size[i], aux_wg_size[i]]])
            data['runtimes'] = torch.tensor([[runtime_cpu[i], runtime_gpu[i]]])
            # TODO hacky fix for 
            if hasattr(data, 'runtime'): delattr(data, 'runtime')

            data_list.append(data)

        ##################################

        print(f" * COMPLETED * === DATASET Devmap-{self.split}: now pre-filtering...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(f" * COMPLETED * === DATASET Devmap-{self.split}: Completed filtering, now pre_transforming...")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100):
            self._save_train_subset()




class POJ104Dataset(InMemoryDataset):
    def __init__(self, root='deeplearning/ml4pl/poj104/classifyapp_data',
                 split='fail',
                 transform=None, pre_transform=None,
                 train_subset=[0, 100], train_subset_seed=0):
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
            print(f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk.")
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), 'shouldnt be'
        processed_path = str(full_dataset)

        # read data into huge `Data` list.
        data_list = []

        ds_base = Path(self.root)
        print(f'Creating {self.split} dataset at {str(ds_base)}')

        split_folder = ds_base / ('ir_' + self.split)
        assert split_folder.exists(), f"{split_folder} doesn't exist!"
        
        # collect .ll.p instead and call nx2data on the fly!
        print(f"=== DATASET {split_folder}: Collecting .ll.p files into dataset")

        # only take files from subfolders (with class names!) recursively
        files = [x for x in split_folder.rglob("*.ll.p") if x.parent.name != split_folder.name]
        for file in tqdm.tqdm(files):
            # skip classes that are larger than what config says to enable debugging with less data
            class_label = int(file.parent.name) - 1  # let classes start from 0.
            if class_label >= num_classes:
                continue

            g = load(file)
            data = nx2data(g, class_label)
            data_list.append(data)

        print(f" * COMPLETED * === DATASET {split_folder}: now pre-filtering...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(f" * COMPLETED * === DATASET {split_folder}: Completed filtering, now pre_transforming...")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100) and self.split not in ['val', 'test']:
            self._save_train_subset()




class LegacyPOJ104Dataset(InMemoryDataset):
    def __init__(self, root='deeplearning/ml4pl/poj104/classifyapp_data',
                 split='fail',
                 transform=None, pre_transform=None,
                 train_subset=[0, 100], train_subset_seed=0):
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
            print(f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk.")
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), 'shouldnt be'
        processed_path = str(full_dataset)

        # read data into huge `Data` list.
        data_list = []

        ds_base = Path(self.root)
        print(f'Creating {self.split} dataset at {str(ds_base)}')
        # TODO change this line to go to the new format
        out_base = ds_base / ('ir_' + self.split + '_programl')
        assert out_base.exists(), f"{out_base} doesn't exist!"
        # TODO collect .ll.pickle instead and call nx2data on the fly!
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


if __name__ == '__main__':
    #d = NewNCCDataset()
    #print(d.data)
    root = '/home/zacharias/llvm_datasets/threadcoarsening_data/'
    a = ThreadcoarseningDataset(root, 'Cypress')
    b = ThreadcoarseningDataset(root, 'Tahiti')
    c = ThreadcoarseningDataset(root, 'Fermi')
    d = ThreadcoarseningDataset(root, 'Kepler')
