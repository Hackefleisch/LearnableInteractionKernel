import torch
from e3nn import o3
from e3nn.nn import BatchNorm
from e3nn.nn import FullyConnectedNet
from e3nn.math import soft_one_hot_linspace
from torch_scatter import scatter

from torch_geometric.data import Data

from prepare_pdbbind import atom_types
from prepare_pdbbind import bond_types

from models.graph_utils import calc_interaction_edges

class e3mp_step(torch.nn.Module):

    def __init__(self, config, is_initial=False):
        super(e3mp_step, self).__init__()

        # Variables
        self.node_embedding_scalars = config['node_embedding_scalars']
        self.batch_normalize_msg = config['batch_normalize_msg']
        self.batch_normalize_node_update = config['batch_normalize_node_upd']
        self.is_initial = is_initial

        self.importance = torch.nn.Parameter(torch.randn(1))

        # irreps
        self.irreps = o3.Irreps( str(config['node_embedding_scalars']) + "x0e + " + 
                                 str(config['node_embedding_vectors']) + "x1o + " + 
                                 str(config['node_embedding_tensors']) + "x2e" 
                            ).remove_zero_multiplicities()
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=config['spherical_harmonics_l'])


        # e3 tensor products
        self.e3tp_message = o3.FullyConnectedTensorProduct(self.irreps,
                                                           self.irreps_sh,
                                                           self.irreps, 
                                                           shared_weights=False)

        # mlps
        # the messages use the edge labels to weight the tensor product
        self.msg_weights_layers = [len(bond_types)]
        self.msg_weights_layers.extend( config['msg_weights_hidden_layers'] )
        self.msg_weights_layers.append( self.e3tp_message.weight_numel )
        self.mlp_message_weights = FullyConnectedNet(self.msg_weights_layers, config['msg_weights_act'])
        
        # the node update uses only the scalar features
        self.node_update_layers = [config['node_embedding_scalars']*2]
        self.node_update_layers.extend( config['node_update_hidden_layers'] )
        self.node_update_layers.append( config['node_embedding_scalars'] )
        self.mlp_node_update = FullyConnectedNet(self.node_update_layers, config['node_update_act'])

        # batch normalizer
        self.bn_msg = BatchNorm(self.irreps) if self.batch_normalize_msg else None
        self.bn_node_upd = BatchNorm(self.irreps) if self.batch_normalize_node_update else None

    def forward(self, graph):
        # generate edge embeddings in spherical harmonics
        edge_src=graph.edge_index[0]
        edge_dst=graph.edge_index[1]
        num_neighbors = edge_src.size(dim=0) / graph.x.size(dim=0)
        edge_vec=graph.pos[edge_dst]-graph.pos[edge_src]
        edge_sh=o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')
        
        # calculate messages
        tp_weights = self.mlp_message_weights(graph.edge_attr)
        tp_results = self.e3tp_message(graph.x[edge_src], edge_sh, tp_weights)
        messages = scatter(tp_results, edge_dst, dim=0, dim_size=graph.x.size(dim=0)).div(num_neighbors**0.5)

        if self.batch_normalize_msg:
            messages = self.bn_msg(messages)

        messages *= self.importance

        # Nodes are updated by summing geometric features and mlps for scalar
        scalars = torch.cat([messages[:,:self.node_embedding_scalars],
                             graph.x[:,:self.node_embedding_scalars]], dim=1)
        scalars = self.mlp_node_update(scalars)
        geoms = messages[:,self.node_embedding_scalars:] + graph.x[:,self.node_embedding_scalars:]
        # node embeddings are initialized with zeros for geometries so there is no need for a division by two
        if not self.is_initial:
            geoms /= 2


        updated_node_features = torch.cat([scalars, geoms], dim=1)

        if self.batch_normalize_node_update:
            updated_node_features = self.bn_node_upd(updated_node_features)

        # update node features
        return updated_node_features

class e3pattern(torch.nn.Module):

    def __init__(self, config):
        super(e3pattern, self).__init__()


        # variables
        self.node_embedding_scalars = config['node_embedding_scalars']
        self.node_embedding_vectors = config['node_embedding_vectors']
        self.node_embedding_tensors = config['node_embedding_tensors']
        self.n_layers = config['n_pattern_layers']

        # irreps
        self.irreps = o3.Irreps( str(self.node_embedding_scalars) + "x0e + " + 
                                 str(self.node_embedding_vectors) + "x1o + " + 
                                 str(self.node_embedding_tensors) + "x2e" 
                            ).remove_zero_multiplicities()

        # mlps
        self.initial_node_emb_layers = [len(atom_types)]
        self.initial_node_emb_layers.extend( config['node_emb_hidden_layers'] )
        self.initial_node_emb_layers.append( self.node_embedding_scalars )
        self.mlp_initial_node_embedder = FullyConnectedNet(self.initial_node_emb_layers, config['node_emb_act'])

        # message passing layers
        self.layers = torch.nn.ModuleList()

        # TODO: Do I want to add a learnable scalar which weights the message in comparison to the node feature?
        #       E.g. to say the first messages are more important than following or vice versa
        self.layers.append(e3mp_step(config, is_initial=True))
        
        # TODO: Add gates!
        for _ in range(1,self.n_layers):
            self.layers.append(e3mp_step(config, is_initial=False))
    
    def forward(self, graph):
        node_emb = self.mlp_initial_node_embedder(graph.x)
        missing_features = self.irreps.dim - self.node_embedding_scalars
        # pad the embedding with zeros for the uninitialized geometric features
        node_emb = torch.cat([node_emb, torch.zeros([node_emb.size(dim=0), missing_features], device=node_emb.device)], dim=1)

        updated_graph = Data(x=node_emb, edge_index=graph.edge_index, pos=graph.pos, edge_attr=graph.edge_attr)
        for layer in self.layers:
            updated_graph.x = layer(updated_graph)

        return updated_graph
    

class e3interaction(torch.nn.Module):

    def __init__(self, config):
        super(e3interaction, self).__init__()

        # variables
        self.radius = config['radius']
        self.num_basis = int(self.radius * config['basis_density_per_A'])
        
        # irreps
        self.irreps_out =  o3.Irreps( str(config['out_scalars']) + "x0e + " + 
                                      str(config['out_vectors']) + "x1o + " + 
                                      str(config['out_tensors']) + "x2e" 
                            ).remove_zero_multiplicities()
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=config['spherical_harmonics_l'])
        self.irreps_node = o3.Irreps( str(config['node_embedding_scalars']) + "x0e + " + 
                                      str(config['node_embedding_vectors']) + "x1o + " + 
                                      str(config['node_embedding_tensors']) + "x2e" 
                            ).remove_zero_multiplicities()

        # e3 tp
        self.e3tp_lig_interaction = o3.FullyConnectedTensorProduct(self.irreps_node, self.irreps_sh, self.irreps_node, shared_weights=False)
        self.e3tp_rec_interaction = o3.FullyConnectedTensorProduct(self.irreps_node, self.irreps_node, self.irreps_out, shared_weights=False)

        # mlps
        self.tp_lig_weights_layers = [self.num_basis]
        self.tp_lig_weights_layers.extend( config['interaction_tp_lig_weights_hidden_layers'] )
        self.tp_lig_weights_layers.append( self.e3tp_lig_interaction.weight_numel )
        self.mlp_lig_weights = FullyConnectedNet(self.tp_lig_weights_layers, config['interaction_tp_lig_weights_act'])

        self.tp_rec_weights_layers = [self.num_basis]
        self.tp_rec_weights_layers.extend( config['interaction_tp_rec_weights_hidden_layers'] )
        self.tp_rec_weights_layers.append( self.e3tp_rec_interaction.weight_numel )
        self.mlp_rec_weights = FullyConnectedNet(self.tp_rec_weights_layers, config['interaction_tp_rec_weights_act'])

    def forward(self, interaction_graph):
        edge_rec = interaction_graph.edge_index[ 0 ]
        edge_lig = interaction_graph.edge_index[ 1 ]

        edge_vec = interaction_graph.pos[edge_lig] - interaction_graph.pos[edge_rec]
        sh_edge_embedding = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')
        
        distance_embedding = soft_one_hot_linspace(edge_vec.norm(dim=1), 0.0, self.radius, self.num_basis, basis='smooth_finite', cutoff=True).mul(self.num_basis**0.5)
        lig_node_embedding = self.e3tp_lig_interaction(interaction_graph.x[edge_lig], sh_edge_embedding, self.mlp_lig_weights(distance_embedding))
        return self.e3tp_rec_interaction(lig_node_embedding, interaction_graph.x[edge_rec], self.mlp_rec_weights(distance_embedding))


   
class InteractionPredictor(torch.nn.Module):

    def __init__(self, config):
        super(InteractionPredictor, self).__init__()

        # variables
        self.radius = config['radius']

        # variable buffer
        self.register_buffer('hydrogen_embedding', torch.zeros(len(atom_types), requires_grad=False))

        # modules
        self.pattern_detector = e3pattern(config)
        self.interaction = e3interaction(config)

    def forward(self, combined_graph, all_inter_edge_index=None):
        
        if all_inter_edge_index == None:
            self.hydrogen_embedding[ atom_types.index('H') ] = 1
            all_inter_edge_index = calc_interaction_edges(combined_graph, self.radius, self.hydrogen_embedding)

        pattern_rec_graph = self.pattern_detector(combined_graph)

        interaction_graph = Data(x=pattern_rec_graph.x, pos=combined_graph.pos, edge_index=all_inter_edge_index)

        return self.interaction(interaction_graph), all_inter_edge_index
    