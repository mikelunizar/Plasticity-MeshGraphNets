from torch_geometric.data import Data
import enum
import numpy as np
import datetime
import json
from pathlib import Path

from amb.metrics import se_inf

VAR = ['COORD.COOR1', 'COORD.COOR2', 'COORD.COOR3', 'S.S11', 'S.S22', 'S.S33', 'S.S12', 'S.S13', 'S.S23', 'S.Mises']


class NodeType(enum.IntEnum):
    NORMAL = 0
    ACTUATOR = 1
    FIXED_POINTS = 3


def set_run_directory(args):
    # Generate a unique name for the run directory based on current timestamp and arguments
    name = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{args.run_name}_layers={args.layers}_MPSteps={args.mp_steps}' \
           f'_hidden={args.hidden}_batchsize={args.batch_size}'

    # Create a Path object for the run directory
    chckp_path = Path(f'outputs/runs/{name}')
    # Create the directory if it doesn't exist, including parent directories
    chckp_path.mkdir(exist_ok=True, parents=True)

    # Save the configuration as JSON in the run directory
    with open(chckp_path / 'config.json', 'w') as jsonfile:
        json.dump(vars(args), jsonfile, indent=4)

    # Return the path to the run directory
    return chckp_path, name


def decompose_graph(graph):
    # graph: torch_geometric.data.data.Data
    return (graph.x, graph.edge_index, graph.edge_attr, graph.edge_world_index, graph.edge_world_attr)


def copy_geometric_data(graph):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """
    node_attr, edge_index, edge_attr, edge_world_index, edge_world_attr = decompose_graph(graph)
    
    ret = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, edge_world_index=edge_world_index, edge_world_attr=edge_world_attr)
    
    return ret


def copy_geometric_graph_data(graph):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """        
    return Data(x=graph.x, y=graph.y, face=graph.face, n=graph.n, pos=graph.pos, mesh_pos=graph.mesh_pos, batch=graph.batch, ptr=graph.ptr)


def compute_errors(predictions, targets, mask_object):
    se_pos = (predictions[:, mask_object, :3] - targets[:, mask_object, :3]) ** 2
    se_stress = (predictions[:, mask_object, 3:] - targets[:, mask_object, 3:]) ** 2
    return np.mean(se_pos), np.mean(se_stress), se_pos.reshape(-1), se_stress.reshape(-1)

def compute_relative_errors(predictions, targets, mask_object, se_inf):
    se_inf_pos = se_inf(targets[:, mask_object, :3], predictions[:, mask_object, :3])
    se_inf_stress = se_inf(targets[:, mask_object, 3:], predictions[:, mask_object, 3:])
    return np.mean(se_inf_pos), np.mean(se_inf_stress), se_inf_pos.reshape(-1), se_inf_stress.reshape(-1)

def compute_statistics(errors):
    rmse = np.sqrt([np.mean(err) for err in errors])
    std_dev = np.std(rmse, ddof=1)
    se = std_dev / np.sqrt(len(errors))

    rmse_total = np.sqrt([np.mean(np.concatenate(errors))])[0]
    return rmse_total, se

def save_results(path, steps, mse_pos_dict, mse_stress_dict, mse_inf_pos_dict, mse_inf_stress_dict):
    data = {
        'metadata': {'steps': steps, 'tra': len(mse_pos_dict)},
        'mse_positions': mse_pos_dict,
        'mse_stresses': mse_stress_dict,
        'mse_inf_positions': mse_inf_pos_dict,
        'mse_inf_stresses': mse_inf_stress_dict
    }
    with open(f"{path}/errors.json", 'a') as json_file:
        json.dump(data, json_file, indent=4)

def log_results(path, steps, num_trajectories, rmse_pos, rmse_stress, rrmse_pos, rrmse_stress, se_rmse_pos, se_rmse_stress, se_rrmse_pos, se_rrmse_stress):
    log_text = (f"ROLLOUT STEPS={steps} for NUM TRAJ={num_trajectories}\n"
                f"    RMSE positions: {rmse_pos}\n"
                f"    RMSE stress: {rmse_stress}\n"
                f"    RRMSE positions: {rrmse_pos}\n"
                f"    RRMSE stress: {rrmse_stress}\n"
                f"    SE of RMSE positions: {se_rmse_pos}\n"
                f"    SE of RMSE stress: {se_rmse_stress}\n"
                f"    SE of RRMSE positions: {se_rrmse_pos}\n"
                f"    SE of RRMSE stress: {se_rrmse_stress}\n")
    print(log_text)
    with open(f"{path}/errors.txt", 'a') as file:
        file.write(log_text)

from types import SimpleNamespace

def load_pretrain_config(args, run_name):
    if args.pretrain is not None:
        pretrain_path = Path(args.pretrain)

        with open(pretrain_path.parent.parent / 'config.json', 'r') as file:
            config = json.load(file)
        args = SimpleNamespace(**config)
        args.run_name = run_name + '_Pretrain_' + args.run_name
    else:
        pretrain_path = None

    layers = args.layers
    hidden = args.hidden
    mp_steps = args.mp_steps

    return layers, hidden, mp_steps, pretrain_path