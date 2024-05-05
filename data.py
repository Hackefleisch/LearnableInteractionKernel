from torch.utils.data import Dataset
import torch

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
        rec_graph = torch.load(path + "_rec.graph")
        lig_graph = torch.load(path + "_lig.graph")
        interactions = torch.load(path + "_" + self.interaction_type + ".tensor")
        return rec_graph, lig_graph, interactions, self.pdbcodes[idx]
    
    def collate_fn(self, data):
        rec_graphs = []
        lig_graphs = []
        interactions = []
        pdb_codes = []
        for rec_g, lig_g, intera, pdb in data:
            rec_graphs.append(rec_g)
            lig_graphs.append(lig_g)
            interactions.append(intera)
            pdb_codes.append(pdb)
        return rec_graphs, lig_graphs, interactions, pdb_codes