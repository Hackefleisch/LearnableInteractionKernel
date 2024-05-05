import torch
from e3nn import o3
from e3nn.nn import Gate
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
                 node_update_hidden_layers, node_update_act):
        super(e3mp_step, self).__init__()
        # irreps
        self.irreps_node_in = irreps_node_in
        self.irreps_message = irreps_message
        self.irreps_node_out = irreps_node_out
        self.irreps_sh = irreps_sh

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

    def forward(self, graph, initial_node_features):
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

        # update node features
        return self.e3tp_node_update(graph.x, messages, self.mlp_node_update_weights(initial_node_features) )

class e3pattern(torch.nn.Module):

    def __init__(self, n_layers,
                 irreps_input, irreps_message, spherical_harmonics_l, irreps_node,
                 node_embedding_size, node_emb_hidden_layers, node_act,
                 edge_embedding_size, edge_emb_hidden_layers, edge_act,
                 msg_weights_hidden_layers, msg_weights_act,
                 node_update_hidden_layers, node_update_act):
        super(e3pattern, self).__init__()


        # variables
        self.node_embedding_size = node_embedding_size
        self.edge_embedding_size = edge_embedding_size
        self.n_layers = n_layers
        self.msg_weights_hidden_layers=msg_weights_hidden_layers
        self.msg_weights_act=msg_weights_act
        self.node_update_hidden_layers=node_update_hidden_layers
        self.node_update_act=node_update_act

        # irreps
        self.irreps_input = irreps_input
        self.irreps_message = irreps_message
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=spherical_harmonics_l)
        self.irreps_node = irreps_node


        # mlps
        self.node_emb_layers = [len(atom_types)]
        self.node_emb_layers.extend( node_emb_hidden_layers )
        self.node_emb_layers.append( self.node_embedding_size )
        self.mlp_node_embedder = FullyConnectedNet(self.node_emb_layers, node_act)
        self.edge_emb_layers = [len(bond_types)]
        self.edge_emb_layers.extend( edge_emb_hidden_layers )
        self.edge_emb_layers.append( self.edge_embedding_size )
        self.mlp_edge_embedder = FullyConnectedNet(self.edge_emb_layers, edge_act)

        # message passing layers
        self.layers = torch.nn.ModuleList()

        self.layers.append(e3mp_step(irreps_node_in=self.irreps_input,
                                     irreps_message=self.irreps_message,
                                     irreps_node_out=self.irreps_node,
                                     irreps_sh=self.irreps_sh,
                                     n_edge_features=self.edge_embedding_size,
                                     n_node_features=self.node_embedding_size,
                                     msg_weights_hidden_layers=self.msg_weights_hidden_layers,
                                     msg_weights_act=self.msg_weights_act,
                                     node_update_hidden_layers=self.node_update_hidden_layers,
                                     node_update_act=self.node_update_act))
        
        # TODO: Add gates!
        for _ in range(1,n_layers):
            self.layers.append(e3mp_step(irreps_node_in=self.irreps_node,
                                         irreps_message=self.irreps_message,
                                         irreps_node_out=self.irreps_node,
                                         irreps_sh=self.irreps_sh,
                                         n_edge_features=self.edge_embedding_size,
                                         n_node_features=self.node_embedding_size,
                                         msg_weights_hidden_layers=self.msg_weights_hidden_layers,
                                         msg_weights_act=self.msg_weights_act,
                                         node_update_hidden_layers=self.node_update_hidden_layers,
                                         node_update_act=self.node_update_act))
    
    def forward(self, graph):
        node_emb = self.mlp_node_embedder(graph.x)
        edge_emb = self.mlp_edge_embedder(graph.edge_attr)

        initial_node_emb = node_emb
        updated_graph = Data(x=node_emb, edge_index=graph.edge_index, pos=graph.pos, edge_attr=edge_emb)
        for layer in self.layers:
            updated_graph.x = layer(updated_graph, initial_node_emb)

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
    
class InteractionPredictor(torch.nn.Module):

    def __init__(self, n_pattern_layers, radius,
                 irreps_input, irreps_message, pattern_spherical_harmonics_l, irreps_node,
                 node_embedding_size, node_emb_hidden_layers, node_act,
                 edge_embedding_size, edge_emb_hidden_layers, edge_act,
                 msg_weights_hidden_layers, msg_weights_act,
                 node_update_hidden_layers, node_update_act,
                 basis_density_per_A, inter_spherical_harmonics_l,
                 inter_tp_weights_hidden_layers, inter_tp_weights_act,
                 irreps_out):
        super(InteractionPredictor, self).__init__()

        # variables
        self.radius = radius

        # variable buffer
        self.register_buffer('hydrogen_embedding', torch.zeros(len(atom_types), requires_grad=False))

        # modules
        self.pattern_detector = e3pattern(n_pattern_layers,
                 irreps_input, irreps_message, pattern_spherical_harmonics_l, irreps_node,
                 node_embedding_size, node_emb_hidden_layers, node_act,
                 edge_embedding_size, edge_emb_hidden_layers, edge_act,
                 msg_weights_hidden_layers, msg_weights_act,
                 node_update_hidden_layers, node_update_act)
        self.interaction = e3interaction(irreps_node, irreps_out, self.radius, 
                                         basis_density_per_A, inter_spherical_harmonics_l, 
                                         inter_tp_weights_hidden_layers, inter_tp_weights_act)

    def forward(self, rec_graph, lig_graph, inter_edge_index=None):

        # safe these for hydrogen detection later
        rec_atom_labels = rec_graph.x
        lig_atom_labels = lig_graph.x

        pattern_rec_graph = self.pattern_detector(rec_graph)
        pattern_lig_graph = self.pattern_detector(lig_graph)

        all_node_emb = torch.cat([pattern_rec_graph.x,
                                pattern_lig_graph.x
                                ], dim=0)
        
        all_pos = torch.cat([pattern_rec_graph.pos, pattern_lig_graph.pos], dim=0)
        
        if inter_edge_index == None:
            # create edge between neighboring atoms
            inter_edge_index = radius_graph(x=all_pos, r=self.radius, loop=False, max_num_neighbors=all_pos.size(0)-1)

            # Filter edges
            n_rec_atoms = pattern_rec_graph.x.size(0)

            # Mask edges where one end is in rec and the other is in lig
            # This ensures every pair interacts only once and the direction is from receptor to ligand
            mask = (inter_edge_index[0] < n_rec_atoms) & (inter_edge_index[1] >= n_rec_atoms)
            inter_edge_index = inter_edge_index[:, mask]

            # create hydrogen masks
            #       I think hydrogens are important for pattern detection, but it might be a good idea to leave them out for the interaction detection since PLIP ignores them as well
            #       but: They can not be removed for plip or they will change the predicted interactions tremendously
            
            self.hydrogen_embedding[ atom_types.index('H') ] = 1
            rec_h_mask = (rec_atom_labels == self.hydrogen_embedding).all(dim=-1)
            lig_h_mask = (lig_atom_labels == self.hydrogen_embedding).all(dim=-1)

            all_h_mask = torch.cat([rec_h_mask, lig_h_mask], dim=0)
            device = ( "cuda" if all_h_mask.is_cuda else "cpu" )
            hydrogen_indices = torch.arange(0,all_pos.size(0), 1, device=device)[all_h_mask]
            rec_non_h_edge_mask = ((hydrogen_indices.view(-1,1) - inter_edge_index[0]) != 0).all(0)
            lig_non_h_edge_mask = ((hydrogen_indices.view(-1,1) - inter_edge_index[1]) != 0).all(0)
            all_non_h_edge_mask = rec_non_h_edge_mask & lig_non_h_edge_mask
            
            inter_edge_index = inter_edge_index[:, all_non_h_edge_mask]

        interaction_graph = Data(x=all_node_emb, pos=all_pos, edge_index=inter_edge_index)

        return self.interaction(interaction_graph), inter_edge_index