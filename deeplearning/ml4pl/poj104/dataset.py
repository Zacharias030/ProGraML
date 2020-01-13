# better dataloader
from pathlib import Path

import tqdm
import torch
from torch_geometric.data import InMemoryDataset

class POJ104Dataset(InMemoryDataset):
    def __init__(self, root, split, transform=None, pre_transform=None):
        self.split = split
        super().__init__(root, transform, pre_transform)

        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])
        else:
            raise ValueError(f"split={split} but has to be one of train, val, test ")

    @property
    def raw_file_names(self):
        return 'classifyapp_data.zip' #['ir_val', 'ir_val_programl']
    
    @property
    def processed_file_names(self):
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']
    
    def download(self):
        # download to self.raw_dir
        print("I cannot yet download data automatically.")
    
    def process(self):
        # hardcoded
        num_classes = 104
        
        # output file
        if self.split == 'train':
            processed_path = self.processed_paths[0]
        elif self.split == 'val':
            processed_path = self.processed_paths[1]
        else:
            processed_path = self.processed_paths[2]

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
        # ...
        
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(f" * COMPLETED * === DATASET {out_base}: Completed filtering, now pre_transforming...")
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), processed_path)
        
    
