import torch
from e3nn import o3
from e3nn.nn import BatchNorm
from e3nn.nn import FullyConnectedNet
from e3nn.math import soft_one_hot_linspace
from torch_scatter import scatter
from torch_cluster import radius_graph

from torch_geometric.data import Data

from prepare_pdbbind import atom_types
from models.InteractionPredictor import e3interaction
from models.graph_utils import calc_interaction_edges


class PointwiseConvolution(torch.nn.Module):
    def __init__(self, config):
        super(PointwiseConvolution, self).__init__()
        
        # variables
        self.node_embedding_size = config['node_embedding_size']
        self.conv_radius = config['conv_radius']
        self.num_basis = int(self.conv_radius * config['basis_density_per_A'])
        self.batch_normalize = config['batch_normalize']

        # irreps
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=config['spherical_harmonics_l'])
        self.irreps_node_in = o3.Irreps(str(self.node_embedding_size) + "x0e")
        self.irreps_node_out = o3.Irreps( str(config['irreps_node_scalars']) + "x0e + " + 
                                      str(config['irreps_node_vectors']) + "x1o + " + 
                                      str(config['irreps_node_tensors']) + "x2e").remove_zero_multiplicities()
        
        # e3 tp
        self.e3tp = o3.FullyConnectedTensorProduct(self.irreps_node_in, self.irreps_sh, self.irreps_node_out, shared_weights=False)

        # mlps
        self.node_emb_layers = [len(atom_types)]
        self.node_emb_layers.extend( config['node_emb_hidden_layers'] )
        self.node_emb_layers.append( self.node_embedding_size )
        self.mlp_node_embedder = FullyConnectedNet(self.node_emb_layers)
        
        self.tp_weights_layers = [self.num_basis]
        self.tp_weights_layers.extend( config['tp_weights_hidden_layers'] )
        self.tp_weights_layers.append( self.e3tp.weight_numel )
        self.mlp_weights = FullyConnectedNet(self.tp_weights_layers, config['tp_weights_act'])

        # batch normalizer
        self.bn = BatchNorm(self.irreps_node_out) if self.batch_normalize else None


    def forward(self, multigraph):

        # onehot to initial node embeddings
        node_emb = self.mlp_node_embedder(multigraph.x)

        updated_graph = Data(x=node_emb, edge_index=multigraph.edge_index, pos=multigraph.pos, edge_attr=multigraph.edge_attr)
        
        all_intra_edge_index = []
        count_all_nodes = 0
        for subgraph_idx in range(len(multigraph.pdb)):

            num_rec_nodes = multigraph.n_rec_nodes[subgraph_idx]
            num_lig_nodes = multigraph.n_lig_nodes[subgraph_idx]
            
            # filter labels and positions
            rec_atom_pos = multigraph.pos[count_all_nodes:count_all_nodes+num_rec_nodes]
            lig_atom_pos = multigraph.pos[count_all_nodes+num_rec_nodes:count_all_nodes+num_rec_nodes+num_lig_nodes]

            rec_intra_edges = radius_graph(rec_atom_pos, self.conv_radius, max_num_neighbors=num_rec_nodes - 1, loop=False)
            lig_intra_edges = radius_graph(lig_atom_pos, self.conv_radius, max_num_neighbors=num_lig_nodes - 1, loop=False)

            rec_intra_edges += count_all_nodes
            lig_intra_edges += count_all_nodes + num_rec_nodes

            count_all_nodes += num_rec_nodes + num_lig_nodes

            all_intra_edge_index.append(rec_intra_edges)
            all_intra_edge_index.append(lig_intra_edges)

        all_intra_edge_index = torch.cat(all_intra_edge_index, dim=1)
        
        edge_src = all_intra_edge_index[ 0 ]
        edge_dst = all_intra_edge_index[ 1 ]
        num_neighbors = edge_src.size(dim=0) / multigraph.x.size(dim=0)

        edge_vec = multigraph.pos[edge_dst] - multigraph.pos[edge_src]
        sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')
        distance_emb = soft_one_hot_linspace(edge_vec.norm(dim=1), 0.0, self.conv_radius, self.num_basis, basis='smooth_finite', cutoff=True).mul(self.num_basis**0.5)
        
        tensor_products = self.e3tp(updated_graph.x[edge_src], sh, self.mlp_weights(distance_emb))
        messages = scatter(tensor_products, edge_dst, dim=0, dim_size=len(multigraph.x)).div(num_neighbors**0.5)

        updated_graph.x = torch.cat([node_emb, messages], dim=1)

        return updated_graph

class InteractionPredictorPointwise(torch.nn.Module):

    def __init__(self, config):
        super(InteractionPredictorPointwise, self).__init__()

        # variables
        self.radius = config['radius']

        # variable buffer
        self.register_buffer('hydrogen_embedding', torch.zeros(len(atom_types), requires_grad=False))

        # modules
        self.pattern_conv = PointwiseConvolution(config)

        self.interaction = e3interaction(self.pattern_conv.irreps_node_in + self.pattern_conv.irreps_node_out, o3.Irreps("1x0e"), self.radius, 
                                         config['basis_density_per_A'], config['spherical_harmonics_l'], 
                                         config['inter_tp_weights_hidden_layers'], config['inter_tp_weights_act'])
        
    def forward(self, combined_graph, all_inter_edge_index=None):
        
        if all_inter_edge_index == None:
            self.hydrogen_embedding[ atom_types.index('H') ] = 1
            all_inter_edge_index = calc_interaction_edges(combined_graph, self.radius, self.hydrogen_embedding)

        pattern_rec_graph = self.pattern_conv(combined_graph)

        interaction_graph = Data(x=pattern_rec_graph.x, pos=combined_graph.pos, edge_index=all_inter_edge_index)

        return self.interaction(interaction_graph), all_inter_edge_index