import torch.nn as nn
from .blocks import NodeProcessorModule, EdgeMeshProcessorModule, EdgeWorldProcessorModule
from src.utils.utils import decompose_graph, copy_geometric_data
from torch_geometric.data import Data


def instantiate_mlp(in_size, hidden_size, out_size, layers=5, lay_norm=True, dropout=0.0):

    module = [nn.Linear(in_size, hidden_size), nn.ReLU()]
    if dropout > 0.:
        module.append(nn.Dropout(dropout))
    for _ in range(layers-2):
        module.append(nn.Linear(hidden_size, hidden_size))
        module.append(nn.ReLU())
        if dropout > 0.:
            module.append(nn.Dropout(dropout))
    module.append(nn.Linear(hidden_size, out_size))

    module = nn.Sequential(*module)

    if lay_norm: return nn.Sequential(module,  nn.LayerNorm(normalized_shape=out_size))

    return module


class Encoder(nn.Module):

    def __init__(self,
                edge_input_size=128,
                node_input_size=128,
                hidden_size=128,
                layers=2,
                dropout=0.0):
        super(Encoder, self).__init__()

        self.emb_encoder = instantiate_mlp(edge_input_size * 2, hidden_size, hidden_size, layers=layers, dropout=dropout)
        self.ewb_encoder = instantiate_mlp(edge_input_size, hidden_size, hidden_size, layers=layers, dropout=dropout)

        self.nb_encoder = instantiate_mlp(node_input_size, hidden_size, hidden_size, layers=layers, dropout=dropout)
    
    def forward(self, graph):

        node_attr, _, edge_mesh_attr, _, edge_world_attr = decompose_graph(graph)
        node_ = self.nb_encoder(node_attr)
        edge_mesh_ = self.emb_encoder(edge_mesh_attr)
        edge_world_ = self.ewb_encoder(edge_world_attr)
        
        return Data(x=node_, edge_attr=edge_mesh_, edge_index=graph.edge_index, edge_world_index=graph.edge_world_index, edge_world_attr=edge_world_)


class ProcessorMPNN(nn.Module):

    def __init__(self, hidden_size=128, layers=2, dropout=0.0):

        super(ProcessorMPNN, self).__init__()

        eb_input_dim = 3 * hidden_size
        nb_input_dim = 3 * hidden_size

        nb_custom_func = instantiate_mlp(nb_input_dim, hidden_size, hidden_size, layers=layers, dropout=dropout)
        ewb_custom_func = instantiate_mlp(eb_input_dim, hidden_size, hidden_size, layers=layers, dropout=dropout)
        emb_custom_func = instantiate_mlp(eb_input_dim, hidden_size, hidden_size, layers=layers, dropout=dropout)

        self.emb_module = EdgeMeshProcessorModule(model=emb_custom_func)
        self.ewb_module = EdgeWorldProcessorModule(model=ewb_custom_func)
        self.nb_module = NodeProcessorModule(model=nb_custom_func)

    def forward(self, graph):
    
        graph_last = copy_geometric_data(graph)

        graph = self.emb_module(graph)
        graph = self.ewb_module(graph)
        graph = self.nb_module(graph)

        edge_attr = graph_last.edge_attr + graph.edge_attr
        edge_world_attr = graph_last.edge_world_attr + graph.edge_world_attr
        x = graph_last.x + graph.x

        return Data(x=x, edge_attr=edge_attr, edge_world_attr=edge_world_attr, edge_index=graph.edge_index, edge_world_index=graph.edge_world_index)


class Decoder(nn.Module):

    def __init__(self, hidden_size=128, output_size=2, layers=2, dropout=0.0):
        super(Decoder, self).__init__()
        self.decode_module =  instantiate_mlp(hidden_size, hidden_size, output_size, layers=layers, lay_norm=False)

    def forward(self, graph):
        out = self.decode_module(graph.x)
        return out


class EncoderProcessorDecoder(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, output_size=3, layers=2, hidden_size=128, shared_mp=False, dropout=0.0):

        super(EncoderProcessorDecoder, self).__init__()
        # Encoders
        self.encoder = Encoder(edge_input_size=edge_input_size, node_input_size=node_input_size, hidden_size=hidden_size, layers=layers)
        # Processor
        processor_list = [ProcessorMPNN(hidden_size=hidden_size, layers=layers) for _ in range(message_passing_num)]
        self.processer_list = nn.ModuleList(processor_list)
        # Deocder
        self.decoder = Decoder(hidden_size=hidden_size, output_size=output_size, layers=layers)

    def forward(self, graph):
        # Encode
        graph = self.encoder(graph)
        # Process
        for i, model in enumerate(self.processer_list):
            graph = model(graph)
        # Decode
        decoded = self.decoder(graph)
        return decoded







