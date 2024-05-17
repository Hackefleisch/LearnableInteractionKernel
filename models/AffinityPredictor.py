import torch
import torch.nn as nn
from torch_scatter import scatter


class AffinityPredictor(nn.Module):

    def __init__(self, config):

        super(AffinityPredictor, self).__init__()

        self.n_interaction_categories = config['n_interaction_categories']
        self.interaction_embedding_size = config['interaction_embedding_size']

        self.transform_edges = nn.Linear(self.n_interaction_categories, self.interaction_embedding_size) if self.interaction_embedding_size != 0 else None
        self.transform_act = config['transform_act']

        self.affinity_act = config['affinity_act']

        self.calc_affinity = nn.Linear(self.interaction_embedding_size*2 if self.interaction_embedding_size != 0 else self.n_interaction_categories*2, 1) 

    def forward(self, interaction_graph):

        # TODO: Maybe I should normalize here, since hydrogen bonds will report only 2 atom pairs, but pi stacking 6*6=36
        edge_embeddings = self.transform_act( self.transform_edges( interaction_graph.interaction_edge_labels ) ) if self.transform_edges != None else interaction_graph.interaction_edge_labels

        edge_sum = scatter(edge_embeddings.T, interaction_graph.graph_index.T).T
        # TODO: The mean is stupid - the presence of an pi stacking will dilute all other values. 
        #       Or its not because I really should use an embedding size on the one-hots
        edge_mean = edge_sum / interaction_graph.n_interaction_edges
        edge_embeddings = torch.cat([edge_sum,edge_mean], dim=1)

        affinity = self.calc_affinity(edge_embeddings)

        if self.affinity_act != None:
            affinity = self.affinity_act(affinity)

        return affinity