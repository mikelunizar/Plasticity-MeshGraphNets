from src.model.model import EncoderProcessorDecoder
from src.utils.utils import NodeType
from src.utils import normalization

import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

import os
import torch
import wandb


class PlasticitySolver(pl.LightningModule):

    def __init__(self, message_passing_num=15, node_input_size=7, edge_input_size=4, output_size=4, hidden=128, layers=2, model_dir='checkpoint/simulator.pth',
                 transforms=None, noise=3e-3, shared_mp=False, epochs=1, lr=1e-4, dropout=0.0, device='cpu'):
        super().__init__()

        self.model_dir = model_dir

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.output_size = output_size

        self.model = EncoderProcessorDecoder(message_passing_num=message_passing_num, hidden_size=hidden,
                                             node_input_size=self.node_input_size, edge_input_size=self.edge_input_size,
                                             shared_mp=shared_mp, output_size=self.output_size, layers=layers, dropout=dropout)

        self._output_vel_normalizer = normalization.Normalizer(size=3, name='output_vel_normalizer', device=device)
        self._output_stress_normalizer = normalization.Normalizer(size=int(self.output_size - 3), name='output_stress_normalizer', device=device)

        self._node_normalizer = normalization.Normalizer(size=3, name='node_normalizer', device=device)
        self._edge_attr_normalizer = normalization.Normalizer(size=edge_input_size * 2, name='edge_normalizer', device=device)
        self._edge_world_normalizer = normalization.Normalizer(size=edge_input_size, name='edge_world_normalizer', device=device)

        self.transforms = transforms
        self.noise = noise
        self.shared_mp = shared_mp
        self.epochs = epochs
        self.lr = lr

        print('Plasticity Solver Instantiated!')

    def forward(self, graph):

        if self.training and (self.noise > 0):
            graph = self.__add_noise_positions(graph)

        # Compute the Edge attributes for mesh and world
        graph = self.transforms(graph.cpu()).to(self.device)

        # extract data from graph
        node_type = graph.n
        cur_position = graph.pos
        target_position = graph.y[:, :3]
        target_stress = graph.y[:, -1].unsqueeze(-1) if self.output_size == 4 else graph.y[:, 3:]

        # prepare target data
        target_vel = self.__position_to_velocity(target_position, cur_position)
        target_vel_norm = self._output_vel_normalizer(target_vel, self.training)
        target_stress_norm = self._output_stress_normalizer(target_stress, self.training)
        target_vel_stress_norm = torch.cat((target_vel_norm, target_stress_norm), dim=-1)

        # compute velocity with target for actuator
        graph.x = self.__process_node_attr(node_type, target_position, cur_position)
        # Process the attribute data, normalize
        graph.edge_attr, graph.edge_world_attr = self.__process_edge_attr(graph.edge_attr, graph.edge_world_attr)

        # Forward pass through GNN
        predicted_vel_stress_norm = self.model(graph)

        # split output into velocity and stress
        predicted_vel_norm = predicted_vel_stress_norm[:, :3]
        predicted_stress_norm = predicted_vel_stress_norm[:, -1].unsqueeze(-1) if self.output_size == 4 else predicted_vel_stress_norm[:, 3:]

        # prepare predicted data
        predicted_vel = self._output_vel_normalizer.inverse(predicted_vel_norm)
        predicted_stress = self._output_stress_normalizer.inverse(predicted_stress_norm)

        predicted_position = cur_position + predicted_vel
        predicted_position_stress = torch.cat((predicted_position, predicted_stress), dim=-1)

        target_position_stress = graph.y

        return predicted_vel_stress_norm, target_vel_stress_norm, predicted_position_stress, target_position_stress

    def training_step(self, batch, batch_idx):

        graph = batch

        mask_loss = torch.logical_or(batch.n == NodeType.NORMAL, batch.n == NodeType.FIXED_POINTS).squeeze()

        predicted_vel_stress_norm, target_vel_stress_norm, predicted_position_stress, target_position_stress = self.forward(
            graph)

        error = torch.sum((target_vel_stress_norm - predicted_vel_stress_norm) ** 2, dim=1)
        loss = torch.mean(error[mask_loss])
        
        self.log('loss_train', loss.detach().item(), prog_bar=True, on_epoch=True, on_step=False,
            batch_size=error[mask_loss].shape[0])

        return loss

    def validation_step(self, batch, batch_idx):

        graph = batch

        mask_loss = torch.logical_or(batch.n == NodeType.NORMAL, batch.n == NodeType.FIXED_POINTS).squeeze()

        predicted_vel_stress_norm, target_vel_stress_norm, predicted_position_stress, target_position_stress = self.forward(
            graph)
        
        error = torch.sum((target_vel_stress_norm - predicted_vel_stress_norm) ** 2, dim=1)
        loss = torch.mean(error[mask_loss])

        self.log('loss_val', loss.detach().item(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=error[mask_loss].shape[0])

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6),
            'monitor': 'train_loss'}

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    @torch.no_grad()
    def rollout(self, model, loader, with_edges=False, max_steps=None):
        mask_bc_actuator = None
        predicted_position = None

        predicteds = []
        targets = []
        edges = []

        graphs = [graph for graph in loader]
        graph_last = graphs[-1]
        graph_last.y = torch.clone(graph_last.x)

        for i in range(max_steps):

            if (max_steps is not None) and (i > max_steps):
                break

            if i <= (len(graphs) - 1):
                graph = graphs[i]
            else:
                graph = graph_last

            if predicted_position is not None:
                graph.x[:, :3] = predicted_position
                graph.pos = predicted_position

            if mask_bc_actuator is None:
                node_type = graph.n
                mask_bc_actuator = torch.where(node_type != NodeType.NORMAL, True, False).squeeze().to(self.device)
                predicteds.append(graph.x.detach().cpu().numpy())
                targets.append(graph.x.detach().cpu().numpy())

            _, _, prediction, _ = model.forward(graph)  # return pred_vel_stress_norm, tar_vel_stress_norm, pred_pos_stress, tar_pos_stress

            next_position_stress = graph.y.to(self.device)

            predicted_position_stress = prediction
            predicted_position_stress[mask_bc_actuator] = next_position_stress[mask_bc_actuator]
            predicted_position = predicted_position_stress[:, :3]

            predicteds.append(predicted_position_stress.detach().cpu().numpy())
            targets.append(next_position_stress.detach().cpu().numpy())

        result = [np.stack(predicteds), np.stack(targets)]
        n = graph.n.squeeze().cpu().detach().numpy()
        edges = edges if with_edges else None
        face = graph.face_mesh.cpu().numpy() if 'face_mesh' in graph.keys() else graph.face.cpu().numpy()

        return result, n, edges, face

    def __add_noise_positions(self, graph):

        pos = graph.pos
        x = graph.x[:, :3]
        type = graph.n.squeeze()
        # sample random noise
        noise = torch.normal(std=self.noise, mean=0.0, size=pos.shape).to(self.device)
        # Do not apply noise to NON-Normal nodes
        mask_not_normal_node = torch.argwhere(type != NodeType.NORMAL).squeeze()
        noise[mask_not_normal_node] = 0
        noise_mask = noise.to(self.device)
        # add noise to pos and x
        graph.pos = pos + noise_mask  # edge attributes are computed with pos
        graph.x[:, :3] = x + noise_mask

        return graph

    def __process_node_attr(self, types, target_pos, curr_pos):

        node_feature = []
        # build one-hot for node type
        node_type = torch.squeeze(types.long())
        one_hot = F.one_hot(node_type, 4)
        # compute velocity based on future position xt+1
        velocity = target_pos - curr_pos
        # set to zero non-actutor velocities
        mask_no_actuator = torch.argwhere(node_type != NodeType.ACTUATOR).squeeze()
        velocity[mask_no_actuator] = 0.
        velocity = self._node_normalizer(velocity, self.training)

        # append and concatenate node attributes
        node_feature.append(velocity)
        node_feature.append(one_hot)

        node_feats = torch.cat(node_feature, dim=1).float()

        return node_feats

    def __process_edge_attr(self, edge_attr, edge_world_attr):

        edge_attr_norm = self._edge_attr_normalizer(edge_attr, self.training)
        edge_world_attr_norm = self._edge_world_normalizer(edge_world_attr, self.training)

        return edge_attr_norm, edge_world_attr_norm

    def __position_to_velocity(self, target_pos, pos):

        velocity_next = target_pos - pos

        return velocity_next

    def freeze_layers(self):
        # Freeze encoder
        params_encoder = self.model.encoder.parameters()
        params_processor = self.model.processer_list.parameters()

        # Freeze encoder parameters
        for i, param in enumerate(params_encoder):
            param.requires_grad = False
        # Freeze processor parameters
        for i, param in enumerate(params_processor):
            param.requires_grad = False

    def load_checkpoint(self, ckpdir=None):
        if ckpdir is None:
            ckpdir = self.model_dir
        dicts = torch.load(ckpdir, map_location=torch.device(self.device))
        self.load_state_dict(dicts['model'])

        keys = list(dicts.keys())
        keys.remove('model')

        for k in keys:
            v = dicts[k]
            for para, value in v.items():
                object = eval('self.' + k)
                if isinstance(value, int) or isinstance(value, str):
                    setattr(object, para, value)
                else:
                    setattr(object, para, value.to(self.device))

        print('Solver model loaded successfully!')

    def load_blockbone_checkpoint(self, ckpdir=None):
        if ckpdir is None:
            ckpdir = self.model_dir
        dicts = torch.load(ckpdir)

        blockbone = {}
        for k, v in dicts['model'].items():
            if 'decoder' in k:
                blockbone[k] = self.state_dict()[k]
            else:
                blockbone[k] = v

        self.load_state_dict(blockbone)

        print("Simulator model loaded checkpoint %s" % ckpdir)

    def save_checkpoint(self, savedir=None, with_wandb=True):
        if savedir is None:
            savedir = self.model_dir

        os.makedirs(os.path.dirname(savedir), exist_ok=True)

        model = self.state_dict()
        _output_vel_normalizer = self._output_vel_normalizer.get_variable()
        _node_normalizer = self._node_normalizer.get_variable()
        _output_stress_normalizer = self._output_stress_normalizer.get_variable()
        _edge_attr_normalizer = self._edge_attr_normalizer.get_variable()
        _edge_world_normalizer = self._edge_world_normalizer.get_variable()

        to_save = {'model': model, '_output_vel_normalizer': _output_vel_normalizer, '_output_stress_normalizer': _output_stress_normalizer, 
                   '_node_normalizer': _node_normalizer, '_edge_attr_normalizer': _edge_attr_normalizer, 
                   '_edge_world_normalizer': _edge_world_normalizer}

        torch.save(to_save, savedir + '.pth')
        print('Simulator model saved at %s' % savedir)
        
        if with_wandb == True:
            wandb_dir = wandb.run.dir + f'/{savedir.split("/")[-1]}.pth'
            torch.save(to_save, wandb_dir)
            wandb.save(wandb_dir)

    def reset_normalization(self):
        self._output_vel_normalizer = normalization.Normalizer(size=3, name='output_vel_normalizer', device=self.device)
        self._output_stress_normalizer = normalization.Normalizer(size=int(self.output_size - 3),
                                                                  name='output_stress_normalizer', device=self.device)

        self._node_normalizer = normalization.Normalizer(size=3, name='node_normalizer', device=self.device)
        self._edge_attr_normalizer = normalization.Normalizer(size=self.edge_input_size * 2, name='edge_normalizer',
                                                              device=self.device)
        self._edge_world_normalizer = normalization.Normalizer(size=self.edge_input_size, name='edge_world_normalizer',
                                                               device=self.device)

