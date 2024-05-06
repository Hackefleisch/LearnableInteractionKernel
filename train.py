import torch
import numpy as np
import pandas as pd

from torch import nn
from data import PDBBindInteractionDataset
from torch.utils.data import DataLoader
from model import InteractionPredictor
from e3nn import o3

import warnings
warnings.filterwarnings("ignore")


def edge_interactions(interaction_tensor, edges, pdb):
    num_edges = edges.size(1)
    num_interactions = interaction_tensor.size(0)

    edges_expanded = edges.T.unsqueeze(1).expand(-1,num_interactions,-1)
    interactions_expanded = interaction_tensor.unsqueeze(0).expand(num_edges,-1,-1)
    edge_results = (edges_expanded == interactions_expanded).all(-1).any(-1)

    return edge_results.int()

def confusion_matrix_calc(predicted_interactions, true_interactions):
    predicted_interactions_class = (torch.sigmoid(predicted_interactions) > 0.85).long()
    true_positive_mask = true_interactions == 1
    true_negative_mask = true_interactions != 1

    true_positives_predictions = (predicted_interactions_class[true_positive_mask] == true_interactions[true_positive_mask]).type(torch.int).sum().item()
    true_negatives_predictions = (predicted_interactions_class[true_negative_mask] == true_interactions[true_negative_mask]).type(torch.int).sum().item()

    positive_predictions_size = true_positive_mask.int().sum().item()
    negative_predictions_size = true_negative_mask.int().sum().item()

    return torch.tensor([[true_positives_predictions, negative_predictions_size - true_negatives_predictions],
                         [positive_predictions_size - true_positives_predictions, true_negatives_predictions]], dtype=torch.int)

def interaction_epoch(dataloader, interaction_model, loss_fn, optimizer, device):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    interaction_model.train()
    total_loss = 0
    confusion_matrix = torch.zeros( (2,2), dtype=torch.int )
    for multigraph in dataloader:
        multi_g = multigraph.to(device)

        # Compute prediction and loss
        predicted_interactions, edges = interaction_model(multi_g)
        predicted_interactions = predicted_interactions.squeeze()

        true_interactions = edge_interactions(multi_g.y, edges, multi_g.pdb)
        loss = loss_fn(predicted_interactions, true_interactions.float())

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        confusion_matrix += confusion_matrix_calc( predicted_interactions, true_interactions )

    return total_loss, confusion_matrix

def interaction_eval(dataloader, interaction_model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    interaction_model.eval()
    total_loss = 0
    confusion_matrix = torch.zeros( (2,2), dtype=torch.int )

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for multigraph in dataloader:
            multi_g = multigraph.to(device)

            predicted_interactions, edges = interaction_model(multi_g)
            predicted_interactions = predicted_interactions.squeeze()

            true_interactions = edge_interactions(multi_g.y, edges, multi_g.pdb)

            total_loss += loss_fn(predicted_interactions, true_interactions.float()).item()
            confusion_matrix += confusion_matrix_calc( predicted_interactions, true_interactions )

    return total_loss, confusion_matrix

def train(num_epochs, eval_every_n_epochs, dataloader_train, dataloader_eval, interaction_model, loss_fn, optimizer, save_weights=True):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device.")
    interaction_model.to(device)
    best_eval_loss = 99999999.9
    for e in range(1, num_epochs+1):
        epoch_loss, confusion_matrix = interaction_epoch(dataloader_train, interaction_model, loss_fn, optimizer, device)
        tp = confusion_matrix[0,0]
        fn = confusion_matrix[1,0]
        fp = confusion_matrix[0,1]
        recall = 100*tp/(tp+fn)
        precision = 0.0
        if tp + fp != 0:
            precision = 100*tp / (tp+fp)
        print(f"loss: {epoch_loss/len(dataloader_train):>7.10f}  recall: {recall:>6.2f}%  precision: {precision:>6.2f}%  [{e:>5d}/{num_epochs:>5d}]")
        if e % eval_every_n_epochs == 0 or e == num_epochs:
            eval_loss, eval_conf_matrix = interaction_eval(dataloader_eval, interaction_model, loss_fn, device)
            tp = eval_conf_matrix[0,0]
            fn = eval_conf_matrix[1,0]
            fp = eval_conf_matrix[0,1]
            recall = 100*tp/(tp+fn)
            precision = 0.0
            if tp + fp != 0:
                precision = 100*tp / (tp+fp)
            print(f"####    eval set loss: {eval_loss/len(dataloader_eval):>7.10f}  recall: {recall:>6.2f}%  precision: {precision:>6.2f}%")
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                if save_weights:
                    torch.save(interaction_model.state_dict(), "model_weights/model_" + str(e) + ".weights")


def main():

    failed_pdb_list = []

    df = pd.read_csv('LP_PDBBind.csv', index_col=0)
    train_pdbs = [ x for x in list(df[(df['new_split'] == 'train') & df.CL1 & ~df.covalent].index) if x not in failed_pdb_list ]
    test_pdbs = [ x for x in list(df[(df['new_split'] == 'test') & df.CL1 & ~df.covalent].index) if x not in failed_pdb_list ]

    pdb_list = []

    with open("INDEX_structure.2020") as file:
        for line in file:
            line=line.strip()
            if line[0] != "#" and line != "":
                pdb_list.append(line.split()[0])
    

    errorlist = ['1b6j', '6rsa']
    pdb_list = [ pdb for pdb in pdb_list if pdb not in errorlist]

    # these are available after recalc
    train_pdbs = pdb_list[:200]
    test_pdbs = pdb_list[200:300]

    for interaction_type in ['hbond']:
        print("@@@@ ", interaction_type)
        inter_pred = InteractionPredictor(n_pattern_layers = 3, 
                                    radius = 7.5,
                                    irreps_input = o3.Irreps("8x0e"),
                                    irreps_message = o3.Irreps("8x0e + 1x1o + 1x2e"),
                                    pattern_spherical_harmonics_l = 2,
                                    irreps_node = o3.Irreps("8x0e + 1x1o + 1x2e"),
                                    node_embedding_size = 8, 
                                    node_emb_hidden_layers = [], 
                                    node_act = torch.relu,
                                    edge_embedding_size = 4,
                                    edge_emb_hidden_layers = [],
                                    edge_act = torch.relu,
                                    msg_weights_hidden_layers = [16],
                                    msg_weights_act = torch.relu,
                                    node_update_hidden_layers = [24], 
                                    node_update_act = torch.relu,
                                    basis_density_per_A = 5, 
                                    inter_spherical_harmonics_l = 2,
                                    inter_tp_weights_hidden_layers = [24], 
                                    inter_tp_weights_act = torch.relu,
                                    irreps_out = o3.Irreps("1x0e"),
                                    batch_normalize_update=False,
                                    batch_normalize_msg=True)

        print("Model weights:", sum(p.numel() for p in inter_pred.parameters() if p.requires_grad))

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(inter_pred.parameters(), lr=1e-3, amsgrad=True)

        dataset_train = PDBBindInteractionDataset("pdbbind2020/", train_pdbs, interaction_type)
        dataloader_train = DataLoader(dataset_train, batch_size=50, shuffle=True, collate_fn=dataset_train.collate_fn, pin_memory=True, num_workers=10)
        dataset_test = PDBBindInteractionDataset("pdbbind2020/", test_pdbs, interaction_type)
        dataloader_test = DataLoader(dataset_test, batch_size=50, shuffle=True, collate_fn=dataset_test.collate_fn, pin_memory=True, num_workers=10)

        train(600, 100, dataloader_train, dataloader_test, inter_pred, loss_fn, optimizer, save_weights=False)
        print("\n-------------------------------------------------------------\n")

if __name__ == "__main__":
    main()