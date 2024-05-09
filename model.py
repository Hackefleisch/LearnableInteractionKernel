import torch
from e3nn import o3
from e3nn.nn import BatchNorm
from e3nn.nn import FullyConnectedNet
from e3nn.math import soft_one_hot_linspace
from torch_scatter import scatter
from torch_cluster import radius_graph

from torch_geometric.data import Data

from prepare_pdbbind import atom_types
from prepare_pdbbind import bond_types

class e3mp_step(torch.nn.Module):

    def __init__(self, irreps_node_in, irreps_message, irreps_node_out, irreps_sh, 
                 n_edge_features, n_node_features, 
                 msg_weights_hidden_layers, msg_weights_act, 
                 node_update_hidden_layers, node_update_act,
                 batch_normalize_msg, batch_normalize_update):
        super(e3mp_step, self).__init__()
        # irreps
        self.irreps_node_in = irreps_node_in
        self.irreps_message = irreps_message
        self.irreps_node_out = irreps_node_out
        self.irreps_sh = irreps_sh
        self.batch_normalize_msg = batch_normalize_msg
        self.batch_normalize_update = batch_normalize_update

        # e3 tensor products
        self.e3tp_message = o3.FullyConnectedTensorProduct(self.irreps_node_in,
                                                           self.irreps_sh,
                                                           self.irreps_message, 
                                                           shared_weights=False)
        self.e3tp_node_update = o3.FullyConnectedTensorProduct(self.irreps_node_in,
                                                               self.irreps_message, 
                                                               self.irreps_node_out,
                                                               shared_weights=False)

        # mlps
        # the messages use the edge labels to weight the tensor product
        self.msg_weights_layers = [n_edge_features]
        self.msg_weights_layers.extend( msg_weights_hidden_layers )
        self.msg_weights_layers.append( self.e3tp_message.weight_numel )
        self.mlp_message_weights = FullyConnectedNet(self.msg_weights_layers, msg_weights_act)
        
        # the node update uses the original node labels
        self.node_update_layers = [n_node_features]
        self.node_update_layers.extend( node_update_hidden_layers )
        self.node_update_layers.append( self.e3tp_node_update.weight_numel )
        self.mlp_node_update_weights = FullyConnectedNet(self.node_update_layers, node_update_act)

        # batch normalizer
        self.bn_msg = BatchNorm(self.irreps_message) if self.batch_normalize_msg else None
        self.bn_update = BatchNorm(self.irreps_node_out) if self.batch_normalize_update else None

    def forward(self, graph, onehot_node_features):
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

        updated_node_features = self.e3tp_node_update(graph.x, messages, self.mlp_node_update_weights(onehot_node_features) )

        if self.batch_normalize_update:
            updated_node_features = self.bn_update(updated_node_features)

        # update node features
        return updated_node_features

class e3pattern(torch.nn.Module):

    def __init__(self, n_layers,
                 irreps_message, spherical_harmonics_l, irreps_node,
                 node_embedding_size, node_emb_hidden_layers,
                 msg_weights_hidden_layers, msg_weights_act,
                 node_update_hidden_layers, node_update_act,
                 batch_normalize_msg, batch_normalize_update):
        super(e3pattern, self).__init__()


        # variables
        self.node_embedding_size = node_embedding_size
        self.n_layers = n_layers
        self.msg_weights_hidden_layers=msg_weights_hidden_layers
        self.msg_weights_act=msg_weights_act
        self.node_update_hidden_layers=node_update_hidden_layers
        self.node_update_act=node_update_act

        # irreps
        self.irreps_input = o3.Irreps( str(node_embedding_size) + "x0e").remove_zero_multiplicities()
        self.irreps_message = irreps_message
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=spherical_harmonics_l)
        self.irreps_node = irreps_node


        # mlps
        self.node_emb_layers = [len(atom_types)]
        self.node_emb_layers.extend( node_emb_hidden_layers )
        self.node_emb_layers.append( self.node_embedding_size )
        self.mlp_node_embedder = FullyConnectedNet(self.node_emb_layers)

        # message passing layers
        self.layers = torch.nn.ModuleList()

        self.layers.append(e3mp_step(irreps_node_in=self.irreps_input,
                                     irreps_message=self.irreps_message,
                                     irreps_node_out=self.irreps_node,
                                     irreps_sh=self.irreps_sh,
                                     n_edge_features=len(bond_types),
                                     n_node_features=len(atom_types),
                                     msg_weights_hidden_layers=self.msg_weights_hidden_layers,
                                     msg_weights_act=self.msg_weights_act,
                                     node_update_hidden_layers=self.node_update_hidden_layers,
                                     node_update_act=self.node_update_act,
                                     batch_normalize_msg=batch_normalize_msg,
                                     batch_normalize_update=batch_normalize_update))
        
        # TODO: Add gates!
        for _ in range(1,n_layers):
            self.layers.append(e3mp_step(irreps_node_in=self.irreps_node,
                                         irreps_message=self.irreps_message,
                                         irreps_node_out=self.irreps_node,
                                         irreps_sh=self.irreps_sh,
                                         n_edge_features=len(bond_types),
                                         n_node_features=len(atom_types),
                                         msg_weights_hidden_layers=self.msg_weights_hidden_layers,
                                         msg_weights_act=self.msg_weights_act,
                                         node_update_hidden_layers=self.node_update_hidden_layers,
                                         node_update_act=self.node_update_act,
                                         batch_normalize_msg=batch_normalize_msg,
                                         batch_normalize_update=batch_normalize_update))
    
    def forward(self, graph):
        node_emb = self.mlp_node_embedder(graph.x)

        onehot_node_emb = graph.x
        updated_graph = Data(x=node_emb, edge_index=graph.edge_index, pos=graph.pos, edge_attr=graph.edge_attr)
        for layer in self.layers:
            updated_graph.x = layer(updated_graph, onehot_node_emb)

        return updated_graph

class e3interaction(torch.nn.Module):

    def __init__(self, irreps_node_in, irreps_out, radius, basis_density_per_A, spherical_harmonics_l, tp_weights_hidden_layers, tp_weights_act):
        super(e3interaction, self).__init__()

        # variables
        self.radius = radius
        self.num_basis = int(self.radius * basis_density_per_A)
        
        # irreps
        self.irreps_out = irreps_out
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=spherical_harmonics_l)
        self.irreps_node_in = irreps_node_in

        # e3 tp
        self.e3tp_interaction = o3.FullyConnectedTensorProduct(self.irreps_node_in + self.irreps_sh, self.irreps_node_in, self.irreps_out, shared_weights=False)

        # mlps
        self.tp_weights_layers = [self.num_basis]
        self.tp_weights_layers.extend( tp_weights_hidden_layers )
        self.tp_weights_layers.append( self.e3tp_interaction.weight_numel )
        self.mlp_weights = FullyConnectedNet(self.tp_weights_layers, tp_weights_act)

    def forward(self, interaction_graph):
        edge_rec = interaction_graph.edge_index[ 0 ]
        edge_lig = interaction_graph.edge_index[ 1 ]

        edge_vec = interaction_graph.pos[edge_lig] - interaction_graph.pos[edge_rec]
        sh_edge_embedding = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')
        
        lig_node_embedding = torch.cat( [interaction_graph.x[edge_lig], sh_edge_embedding], dim=1 )
        distance_embedding = soft_one_hot_linspace(edge_vec.norm(dim=1), 0.0, self.radius, self.num_basis, basis='smooth_finite', cutoff=True).mul(self.num_basis**0.5)
        return self.e3tp_interaction(lig_node_embedding, interaction_graph.x[edge_rec], self.mlp_weights(distance_embedding))

def calc_interaction_edges(multigraph, radius, hydrogen_emb = None):
    all_inter_edge_index = []
    count_all_nodes = 0
    for subgraph_idx in range(len(multigraph.pdb)):

        num_rec_nodes = multigraph.n_rec_nodes[subgraph_idx]
        num_lig_nodes = multigraph.n_lig_nodes[subgraph_idx]

        # filter labels and positions
        rec_atom_labels = multigraph.x[count_all_nodes:count_all_nodes+num_rec_nodes]
        lig_atom_labels = multigraph.x[count_all_nodes+num_rec_nodes:count_all_nodes+num_rec_nodes+num_lig_nodes]
        all_pos = multigraph.pos[count_all_nodes:count_all_nodes+num_rec_nodes+num_lig_nodes]

        # create edge between neighboring atoms
        inter_edge_index = radius_graph(x=all_pos, r=radius, loop=False, max_num_neighbors=all_pos.size(0)-1)

        # Mask edges where one end is in rec and the other is in lig
        # This ensures every pair interacts only once and the direction is from receptor to ligand
        mask = (inter_edge_index[0] < num_rec_nodes) & (inter_edge_index[1] >= num_rec_nodes)
        inter_edge_index = inter_edge_index[:, mask]

        # create hydrogen masks
        #       I think hydrogens are important for pattern detection, but it might be a good idea to leave them out for the interaction detection since PLIP ignores them as well
        #       but: They can not be removed for plip or they will change the predicted interactions tremendously
        if hydrogen_emb != None:
            rec_h_mask = (rec_atom_labels == hydrogen_emb).all(dim=-1)
            lig_h_mask = (lig_atom_labels == hydrogen_emb).all(dim=-1)

            all_h_mask = torch.cat([rec_h_mask, lig_h_mask], dim=0)
            device = ( "cuda" if all_h_mask.is_cuda else "cpu" )
            hydrogen_indices = torch.arange(0,all_pos.size(0), 1, device=device)[all_h_mask]
            rec_non_h_edge_mask = ((hydrogen_indices.view(-1,1) - inter_edge_index[0]) != 0).all(0)
            lig_non_h_edge_mask = ((hydrogen_indices.view(-1,1) - inter_edge_index[1]) != 0).all(0)
            all_non_h_edge_mask = rec_non_h_edge_mask & lig_non_h_edge_mask
            
            inter_edge_index = inter_edge_index[:, all_non_h_edge_mask]

        inter_edge_index += count_all_nodes
        count_all_nodes += num_rec_nodes + num_lig_nodes

        all_inter_edge_index.append(inter_edge_index)
    return torch.cat(all_inter_edge_index, dim=1)
   
class InteractionPredictor(torch.nn.Module):

    def __init__(self, n_pattern_layers, radius,
                 irreps_message_scalars, irreps_message_vectors, irreps_message_tensors,
                 pattern_spherical_harmonics_l,
                 irreps_node_scalars, irreps_node_vectors, irreps_node_tensors,
                 node_embedding_size, node_emb_hidden_layers,
                 msg_weights_hidden_layers, msg_weights_act,
                 node_update_hidden_layers, node_update_act,
                 basis_density_per_A, inter_spherical_harmonics_l,
                 inter_tp_weights_hidden_layers, inter_tp_weights_act,
                 batch_normalize_msg, batch_normalize_update):
        super(InteractionPredictor, self).__init__()

        # variables
        self.radius = radius
        self.irreps_message = o3.Irreps( str(irreps_message_scalars) + "x0e + " + str(irreps_message_vectors) + "x1o + " + str(irreps_message_tensors) + "x2e").remove_zero_multiplicities()
        self.irreps_node = o3.Irreps( str(irreps_node_scalars) + "x0e + " + str(irreps_node_vectors) + "x1o + " + str(irreps_node_tensors) + "x2e").remove_zero_multiplicities()

        # variable buffer
        self.register_buffer('hydrogen_embedding', torch.zeros(len(atom_types), requires_grad=False))

        # modules
        self.pattern_detector = e3pattern(n_pattern_layers,
                 self.irreps_message, pattern_spherical_harmonics_l, self.irreps_node,
                 node_embedding_size, node_emb_hidden_layers,
                 msg_weights_hidden_layers, msg_weights_act,
                 node_update_hidden_layers, node_update_act,
                 batch_normalize_msg=batch_normalize_msg,
                 batch_normalize_update=batch_normalize_update)
        self.interaction = e3interaction(self.irreps_node, o3.Irreps("1x0e"), self.radius, 
                                         basis_density_per_A, inter_spherical_harmonics_l, 
                                         inter_tp_weights_hidden_layers, inter_tp_weights_act)

    def forward(self, combined_graph, all_inter_edge_index=None):
        
        if all_inter_edge_index == None:
            self.hydrogen_embedding[ atom_types.index('H') ] = 1
            all_inter_edge_index = calc_interaction_edges(combined_graph, self.radius, self.hydrogen_embedding)

        pattern_rec_graph = self.pattern_detector(combined_graph)

        interaction_graph = Data(x=pattern_rec_graph.x, pos=combined_graph.pos, edge_index=all_inter_edge_index)

        return self.interaction(interaction_graph), all_inter_edge_index
    
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