from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pathlib import Path
import torch


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, batch_size, num_workers=0, transforms=None, all_stress=False):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = 'train'
        self.valid_split = 'valid'
        
        self.trajectory_train = []
        self.trajectory_valid = []

        self.transforms = transforms
        self.all_stress = all_stress

    def setup(self, stage=None):

        dataset_dir_train = Path(self.dataset_dir) / self.train_split
        self.train_dataset = []
        for file_pt in dataset_dir_train.glob('**/*.pt'):
            self.trajectory_train.append(file_pt.stem)
            data = torch.load(file_pt)
            for i, s in enumerate(data):
                if self.all_stress:
                    data[i].x = s.x.float()
                    data[i].y = s.y.float()
                else:
                    data[i].x = torch.cat([s.x[:, :3], s.x[:, -1].unsqueeze(-1)], dim=-1).float()
                    data[i].y = torch.cat([s.y[:, :3], s.y[:, -1].unsqueeze(-1)], dim=-1).float()

                data[i].pos = data[i].pos.float()
                data[i].mesh_pos = data[i].mesh_pos.float()
                data[i].face = torch.transpose(data[i].face[:, 1:], 1, 0).long()
                data[i].n = data[i].n.long()
                data[i].num = int(file_pt.stem)
            self.train_dataset += data

        dataset_dir_valid = Path(self.dataset_dir) / self.valid_split
        self.valid_dataset = []
        for file_pt in dataset_dir_valid.glob('**/*.pt'):
            self.trajectory_valid.append(file_pt.stem)
            data = torch.load(file_pt)
            for i, s in enumerate(data):
                if self.all_stress:
                    data[i].x = s.x.float()
                    data[i].y = s.y.float()
                else:
                    data[i].x = torch.cat([s.x[:, :3], s.x[:, -1].unsqueeze(-1)], dim=-1).float()
                    data[i].y = torch.cat([s.y[:, :3], s.y[:, -1].unsqueeze(-1)], dim=-1).float()

                data[i].pos = data[i].pos.float()
                data[i].mesh_pos = data[i].mesh_pos.float()
                data[i].face = torch.transpose(data[i].face[:, 1:], 1, 0).long()
                data[i].n = data[i].n.long()
                data[i].num = int(file_pt.stem)

            self.valid_dataset += data

    def train_dataloader(self, **kwargs):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, **kwargs)
    
    def val_dataloader(self, **kwargs):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, **kwargs)

