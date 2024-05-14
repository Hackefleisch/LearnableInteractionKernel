import torch
from torch_cluster import radius_graph

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