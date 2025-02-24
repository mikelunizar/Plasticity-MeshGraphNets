import pytorch_lightning as pl

from torch_geometric.loader import DataLoader

import wandb
import torch
import numpy as np
from pathlib import Path

from src.utils.utils import VAR
from src.utils.render import make_pyvista_video_single


class ModelSave(pl.Callback):
    def __init__(self, dirpath=None, save_freq=5, with_wandb=False):
        super().__init__()
        self.dirpath = dirpath
        self.save_freq = save_freq
        self.with_wandb = with_wandb

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch % self.save_freq == 0:
            pl_module.save_checkpoint(savedir=self.dirpath+f'/epoch{trainer.current_epoch}', with_wandb=self.with_wandb)


class RolloutCallback(pl.Callback):
    def __init__(self, dataset_dir=None, split='valid', var='S.Mises', trajectory=0, transforms=None, freq=1, all_stress=False, pe=False):
        super().__init__()

        data = torch.load(Path(dataset_dir) / split / f'{trajectory}.pt')
        for i, s in enumerate(data):
            if all_stress:
                data[i].x = s.x.float()
                data[i].y = s.y.float()
            else:
                data[i].x = torch.cat([s.x[:, :3], s.x[:, -1].unsqueeze(-1)], dim=-1).float()
                data[i].y = torch.cat([s.y[:, :3], s.y[:, -1].unsqueeze(-1)], dim=-1).float()
            data[i].pos = data[i].pos.float()
            data[i].mesh_pos = data[i].mesh_pos.float()
            data[i].face = torch.transpose(data[i].face[:, 1:], 1, 0).long()
            data[i].n = data[i].n.long()

        self.loader = DataLoader(data, batch_size=1)
        self.transforms = transforms
        self.freq = freq
        self.trajectory = trajectory
        self.split = split
        self.all_stress = all_stress
        self.var = var

    def render_trajectory(self, loader, path, tra=0, with_edges=False, max_steps=None, var='S.Mises', codec='png'):

        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        var_index = VAR.index(var) if var not in 'S.Mises' else -1

        trajectory, n, edges, faces = self.get_trajectory(loader, with_edges, max_steps=max_steps)
        path_rollout = make_pyvista_video_single(trajectory[1], faces, var_index=var_index,
                                          max_steps=len(trajectory[0]), path=path, output_file=f'{var}_{tra}')
        return path_rollout

    def render_rollout(self, model, loader, path, epoch=0, codec='libx264', with_edges=True, max_steps=None, mp_steps=15):
        
        path = Path(path) / 'videos' / self.split
        path.mkdir(exist_ok=True, parents=True)

        results, n, edges = self.rollout(model, loader, with_edges=with_edges, max_steps=max_steps, mp_steps=mp_steps)
        RolloutCallback.compute_metrics_at_step_times(results, n)
        
        path_rollout = self.make_video_from_results(results, n, epoch, path, var=self.var,
                                                    edges=edges, codec=codec)

        return path_rollout

    def get_trajectory(self, loader, with_edges=False, max_steps=None):

        predicteds = []
        targets = []
        edges = []
        face = None

        for i, graph in enumerate(loader):

            if (max_steps is not None) and (i > max_steps):
                break

            if face is None:
                try:
                    face = graph.cpu().face_mesh.numpy()
                except Exception:
                    face = graph.cpu().face.numpy()

            graph = self.transforms(graph)

            predicteds.append(torch.cat([graph.pos, graph.x[:, 3:]], dim=-1).detach().cpu().numpy())
            targets.append(graph.y.detach().cpu().numpy())

            if with_edges:
                edges.append(graph.edge_world_index.squeeze().cpu().detach().numpy())
            else:
                edges = None

        result = [np.stack(predicteds), np.stack(targets)]

        return result, graph.n.squeeze().cpu().detach().numpy(), edges, face


