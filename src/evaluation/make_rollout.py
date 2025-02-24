from src.model.transforms import FaceToEdgeTethra, RadiusGraphMesh, ContactDistance, MeshDistance
from src.model.simulator import PlasticitySolver
from src.model.callbacks import RolloutCallback
from src.utils.utils import VAR
from src.utils.render import make_pyvista_video_combined

import torch_geometric.transforms as T
from pathlib import Path
import argparse

device = 'cpu'  # test example 10 with 750 snapshot takes mps = 81 sec || cpu  = 125 sec || gpu = ?? sec

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MeshGraph Simulation')

    parser.add_argument('--path', type=str, default="output/plasticity_vmises/weights/model.pth")
    parser.add_argument('--dataset_dir', type=str, default='data/PlasticDeformingPlate', help='Directory containing dataset')
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--steps', type=int, default=450)
    parser.add_argument('--trajectory',  type=int, default=23)
    parser.add_argument('--var', type=str, default="S.Mises")
    parser.add_argument('--stress_tensor', action="store_true")

    args = parser.parse_args()

    steps = args.steps
    split = args.split
    dataset_dir = args.dataset_dir
    output_size = 10 if args.stress_tensor else 4

    # Set Feature Engineering as transform
    transforms_model = T.Compose(
        [FaceToEdgeTethra(remove_faces=False), RadiusGraphMesh(r=0.03), T.Cartesian(norm=False),
         T.Distance(norm=False), MeshDistance(norm=False), ContactDistance(norm=False)])
    # Instantiate model
    solver = PlasticitySolver(message_passing_num=15, node_input_size=7, edge_input_size=4, output_size=output_size,
                              transforms=transforms_model, layers=2, hidden=128,
                              dropout=0., device='cpu')
    # Load trained weights
    solver.to('cpu')
    solver.load_checkpoint(args.path)
    solver.eval()

    # Solve plastic trajectory Inference Rollout
    print(f'Solving trajectory {args.trajectory}')
    rollout = RolloutCallback(dataset_dir=dataset_dir, split=split, trajectory=args.trajectory)
    results, n, edges, faces = solver.rollout(solver, rollout.loader, max_steps=args.steps)
    predictions, targets = results[0], results[1]

    # render solved trajectory
    path_save = Path(args.path).parent.parent / split / 'rollouts'
    path_save.mkdir(exist_ok=True, parents=True)
    output_file = f'{args.var}_{args.trajectory}'

    make_pyvista_video_combined(predictions, targets, faces, var=args.var, max_steps=steps,
                                path=path_save, output_file=output_file)




