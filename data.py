from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data

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
        for combined_graph in data:
            x.append(combined_graph.x)
            pos.append(combined_graph.pos)
            edge_attr.append(combined_graph.edge_attr)
            edge_index.append(combined_graph.edge_index + sum(n_rec_nodes) + sum(n_lig_nodes))
            y.append(combined_graph.y + sum(n_rec_nodes) + sum(n_lig_nodes))
            pdb.extend(combined_graph.pdb)
            n_rec_nodes.extend(combined_graph.n_rec_nodes)
            n_lig_nodes.extend(combined_graph.n_lig_nodes)

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