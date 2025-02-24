from src.model.transforms import FaceToEdgeTethra, RadiusGraphMesh, ContactDistance, MeshDistance
from src.utils.utils import set_run_directory, load_pretrain_config
from src.model.simulator import PlasticitySolver
from src.model.callbacks import ModelSave
from src.dataset.datamodule import DataModule

from pytorch_lightning.callbacks import LearningRateMonitor
import torch_geometric.transforms as T
import pytorch_lightning as pl

import torch
import argparse

pl.seed_everything(42, workers=True)
SAVE_MODEL_FREQ = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plasticity MeshGraph Collusions')

    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Batch size for training')
    parser.add_argument('--mp_steps', type=int, default=15, help='Batch size for training')
    parser.add_argument('--layers', type=int, default=2, help='Layer of my model')
    parser.add_argument('--hidden', type=int, default=128, help='Layer of my model')
    parser.add_argument('--rollout_freq', type=int, default=2, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Radius for RadiusGraphMesh')

    parser.add_argument("--stress_tensor", action="store_true", help="Makes stress as input for the model")
    parser.add_argument('--dataset_dir', type=str, default="./data/plastic_vancouver", help='Directory containing dataset')
    parser.add_argument('--run_name', type=str, default="PlasticitySolver", help='Directory containing dataset')
    parser.add_argument('--pretrain', type=str, default=None,  help='Directory containing dataset')
    parser.add_argument("--finetune", action="store_true", help="Freeze the encoder and processor network")

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    lr = args.lr
    batch_size = args.batch_size
    finetune = args.finetune
    epochs = args.epochs
    rollout_freq = args.rollout_freq
    run_name = args.run_name

    output_size = 10 if args.stress_tensor else 4

    layers, hidden, mp_steps, pretrain_path = load_pretrain_config(args, run_name)

    # Data preparation
    transforms_model = T.Compose([FaceToEdgeTethra(remove_faces=False), RadiusGraphMesh(r=0.003), T.Cartesian(norm=False),
                                  T.Distance(norm=False), MeshDistance(norm=False), ContactDistance(norm=False)])
    data_module = DataModule(dataset_dir=dataset_dir, batch_size=batch_size)

    # Model instantiation & trainer
    solver = PlasticitySolver(message_passing_num=mp_steps, node_input_size=7, edge_input_size=4, output_size=output_size,
                                 hidden=hidden, layers=layers, transforms=transforms_model, epochs=epochs, lr=lr, device=device)

    # Pretrained model freeze Encoder-Processor part
    if pretrain_path is not None:
        solver.to(device)
        solver.load_checkpoint(pretrain_path)
        # Choose to only train the decoder
        if finetune:
            solver.freeze_layers()

    # Callbacks
    chckp_path, name = set_run_directory(args)
    modelsave = ModelSave(dirpath=str(chckp_path / 'models'), save_freq=SAVE_MODEL_FREQ, with_wandb=False)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Trainer setup
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                            max_epochs=epochs,
                            callbacks=[modelsave, lr_monitor],
                            num_sanity_val_steps=0,
                            deterministic=True,
                            check_val_every_n_epoch=1,
                            )
    # fit model
    trainer.fit(solver, datamodule=data_module)