from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data
import pandas as pd
from prepare_pdbbind import defined_interactions

class PDBBindInteractionDataset(Dataset):

    def __init__(self, pdbbind_path, pdbcodes, interaction_type, pdbcode_file=""):
        self.pdbbind_path = pdbbind_path
        self.pdbcodes = pdbcodes
        self.interaction_type = interaction_type

        if pdbcode_file != "":
            self.pdbcodes = []
            with open(pdbcode_file) as file:
                for line in file:
                    line=line.strip()
                    if len(line) == 4:
                        self.pdbcodes.append(line)

    def __len__(self):
        return len(self.pdbcodes)

    def __getitem__(self, idx):
        pdb_code = self.pdbcodes[idx]
        path = self.pdbbind_path + pdb_code + "/" + pdb_code
        combined_graph = torch.load(path + "_combined.graph")
        interactions = torch.load(path + "_" + self.interaction_type + ".tensor")
        combined_graph.y = interactions
        return combined_graph
    
    def collate_fn(self, data):
        x = []
        pos = []
        edge_attr = []
        edge_index = []
        pdb = []
        n_rec_nodes = []
        n_lig_nodes = []
        y = []
        for graph in data:
            x.append(graph.x)
            pos.append(graph.pos)
            edge_attr.append(graph.edge_attr)
            edge_index.append(graph.edge_index + sum(n_rec_nodes) + sum(n_lig_nodes))
            y.append(graph.y + sum(n_rec_nodes) + sum(n_lig_nodes))
            pdb.extend(graph.pdb)
            n_rec_nodes.extend(graph.n_rec_nodes)
            n_lig_nodes.extend(graph.n_lig_nodes)

        multigraph = Data(
            x = torch.cat(x, dim=0),
            pos = torch.cat(pos, dim=0),
            edge_attr = torch.cat(edge_attr, dim=0),
            edge_index = torch.cat(edge_index, dim=1),
            y = torch.cat(y, dim=0),
            pdb = pdb,
            n_rec_nodes = n_rec_nodes,
            n_lig_nodes = n_lig_nodes,
        )

        return multigraph

class InteractionAffinitySet(Dataset):

    def __init__(self, pdbbind_path, pdbcodes, lppdbbind_file):
        self.pdbbind_path = pdbbind_path
        self.pdbcodes = pdbcodes

        df = pd.read_csv(lppdbbind_file, index_col=0)
        self.affinity = df.loc[self.pdbcodes][["value"]].to_dict()['value']

    def __len__(self):
        return len(self.pdbcodes)

    def __getitem__(self, idx):
        pdb_code = self.pdbcodes[idx]
        path = self.pdbbind_path + pdb_code + "/" + pdb_code
        combined_graph = torch.load(path + "_combined.graph")
        interaction_edges = torch.zeros([2,0])
        interaction_edge_labels = torch.zeros([0,len(defined_interactions)])
        n_interaction_edges = 0
        for interaction_idx in range(len(defined_interactions)):
            interaction_type = defined_interactions[interaction_idx]
            interactions = torch.load(path + "_" + interaction_type + ".tensor")
            # counts the number of atom pairs participating in an interaction
            interaction_edges = torch.cat([interaction_edges, interactions.T],dim=1)
            labels = torch.zeros([interactions.size(dim=0), len(defined_interactions)])
            labels[:,interaction_idx] += 1
            interaction_edge_labels = torch.cat([interaction_edge_labels, labels])
            n_interaction_edges += interactions.size(dim=0)

        combined_graph.interaction_edges = interaction_edges
        combined_graph.interaction_edge_labels = interaction_edge_labels
        combined_graph.affinity = torch.tensor([self.affinity[pdb_code]])
        combined_graph.graph_index = torch.zeros([interaction_edge_labels.size(dim=0),1], dtype=torch.int64)
        combined_graph.n_interaction_edges = torch.tensor([n_interaction_edges])

        return combined_graph
    
    def collate_fn(self, data):
        x = []
        pos = []
        edge_attr = []
        edge_index = []
        pdb = []
        n_rec_nodes = []
        n_lig_nodes = []
        interaction_edges = []
        interaction_edge_labels = []
        affinity = []
        graph_index = []
        n_interaction_edges = []
        for graph in data:
            x.append(graph.x)
            pos.append(graph.pos)
            edge_attr.append(graph.edge_attr)
            edge_index.append(graph.edge_index + sum(n_rec_nodes) + sum(n_lig_nodes))
            graph_index.append(graph.graph_index + len(pdb))
            interaction_edges.append(graph.interaction_edges + sum(n_rec_nodes) + sum(n_lig_nodes))
            interaction_edge_labels.append(graph.interaction_edge_labels)
            affinity.append(graph.affinity)
            n_interaction_edges.append(graph.n_interaction_edges)
            pdb.extend(graph.pdb)
            n_rec_nodes.extend(graph.n_rec_nodes)
            n_lig_nodes.extend(graph.n_lig_nodes)

        multigraph = Data(
            x = torch.cat(x, dim=0),
            pos = torch.cat(pos, dim=0),
            edge_attr = torch.cat(edge_attr, dim=0),
            edge_index = torch.cat(edge_index, dim=1),
            pdb = pdb,
            n_rec_nodes = n_rec_nodes,
            n_lig_nodes = n_lig_nodes,
            interaction_edges = torch.cat(interaction_edges, dim=1),
            interaction_edge_labels = torch.cat(interaction_edge_labels, dim=0),
            affinity = torch.stack(affinity, dim=0),
            n_interaction_edges =  torch.stack(n_interaction_edges, dim=0),
            graph_index = torch.cat(graph_index, dim=0),
        )

        return multigraph