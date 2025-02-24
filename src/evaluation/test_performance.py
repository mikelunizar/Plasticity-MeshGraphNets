from pathlib import Path
from amb.metrics import se_inf
import numpy as np
import torch_geometric.transforms as T

from src.model.simulator import PlasticitySolver
from src.model.transforms import FaceToEdgeTethra, RadiusGraphMesh, ContactDistance, MeshDistance
from src.utils.utils import NodeType, compute_errors, compute_relative_errors, compute_statistics, log_results
from src.model.callbacks import RolloutCallback
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MeshGraph Simulation')

    parser.add_argument('--path', type=str, default="output/plasticity_vmises/weights/model.pth")
    parser.add_argument('--dataset_dir', type=str, default='data/PlasticDeformingPlate', help='Directory containing dataset')
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--steps', type=int, default=450)
    parser.add_argument('--stress_tensor', action="store_true")

    args = parser.parse_args()

    steps = args.steps
    split = args.split
    dataset_dir = args.dataset_dir
    output_size = 10 if args.stress_tensor else 4

    # Set Feature Engineering as transform
    transforms_model = T.Compose([FaceToEdgeTethra(remove_faces=False), RadiusGraphMesh(r=0.03), T.Cartesian(norm=False),
                                  T.Distance(norm=False), MeshDistance(norm=False), ContactDistance(norm=False)])
    # Instantiate model
    solver = PlasticitySolver(message_passing_num=15, node_input_size=7, edge_input_size=4, output_size=output_size,
                            transforms=transforms_model, layers=2, hidden=128,
                            dropout=0., device='cpu')
    # Load trained weights
    solver.to('cpu')
    solver.load_checkpoint(args.path)
    solver.eval()

    # Perform inference and report errors
    mse_pos_dict, mse_stress_dict, mse_inf_pos_dict, mse_inf_stress_dict = {}, {}, {}, {}
    global_se_pos, global_se_stress, global_se_inf_pos, global_se_inf_stress = [], [], [], []

    path_save = Path(args.path).parent.parent / args.split
    path_save.mkdir(exist_ok=True, parents=True)
    path_save = str(path_save)

    for tra_name in (Path(args.dataset_dir) / args.split).iterdir():
        tra = tra_name.stem
        print(f'Solving trajectory {tra}')
        rollout = RolloutCallback(dataset_dir=args.dataset_dir, split=args.split,
                                  trajectory=tra, all_stress=args.stress_tensor)
        results, n, _, _ = solver.rollout(solver, rollout.loader, max_steps=args.steps)
        mask_object = np.argwhere(n != NodeType.ACTUATOR).reshape(-1)

        value_pos, value_stress, se_pos, se_stress = compute_errors(results[0], results[1], mask_object)
        global_se_pos.append(se_pos)
        global_se_stress.append(se_stress)
        mse_pos_dict[f'trajectory_{tra}'] = value_pos
        mse_stress_dict[f'trajectory_{tra}'] = value_stress
        print(f'mse pos = {value_pos}, mse stress = {value_stress}')

        value_inf_pos, value_inf_stress, se_inf_pos, se_inf_stress = compute_relative_errors(results[0], results[1],
                                                                                             mask_object, se_inf)
        global_se_inf_pos.append(se_inf_pos)
        global_se_inf_stress.append(se_inf_stress)
        mse_inf_pos_dict[f'trajectory_{tra}'] = value_inf_pos
        mse_inf_stress_dict[f'trajectory_{tra}'] = value_inf_stress
        print(f'mse inf pos = {value_inf_pos}, mse inf stress = {value_inf_stress}')

    rmse_pos, se_rmse_pos = compute_statistics(global_se_pos)
    rmse_stress, se_rmse_stress = compute_statistics(global_se_stress)
    rrmse_pos, se_rrmse_pos = compute_statistics(global_se_inf_pos)
    rrmse_stress, se_rrmse_stress = compute_statistics(global_se_inf_stress)

    log_results(path_save, args.steps, len(mse_pos_dict), rmse_pos, rmse_stress, rrmse_pos, rrmse_stress,
                se_rmse_pos,
                se_rmse_stress, se_rrmse_pos, se_rrmse_stress)


