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
    interaction_model.train()
    total_loss = 0
    confusion_matrix = torch.zeros( (2,2), dtype=torch.int )
    for multigraph in dataloader:
        multi_g = multigraph.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # Compute prediction and loss
        predicted_interactions, edges = interaction_model(multi_g)
        predicted_interactions = predicted_interactions.squeeze()

        true_interactions = edge_interactions(multi_g.y, edges, multi_g.pdb)
        loss = loss_fn(predicted_interactions, true_interactions.float())

        # Backpropagation
        loss.backward()
        optimizer.step()

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

            total_loss += loss_fn(predicted_interactions, true_interactions.float()).item() if loss_fn != None else 0
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

    return eval_loss

def create_model(config):
    acts = {
        "ReLU" : nn.ReLU(), 
        "Sigmoid" : nn.Sigmoid(), 
        "Hardtanh" : nn.Hardtanh()
    }

    inter_pred = InteractionPredictor(
            # node embedding mlp
            node_emb_hidden_layers = [config['node_emb_hidden_layers']] if config['node_emb_hidden_layers'] != 0 else [],
            node_embedding_size = config['node_embedding_size'],

            # message weights mlp
            msg_weights_hidden_layers = [config['msg_weights_hidden_layers']] if config['msg_weights_hidden_layers'] != 0 else [],
            msg_weights_act = acts[config['weights_act']],

            # message tp spherical harmonics edge
            pattern_spherical_harmonics_l = config['spherical_harmonics_l'],

            # message format
            irreps_message_scalars = config['irreps_message_scalars'], 
            irreps_message_vectors = config['irreps_message_vectors'], 
            irreps_message_tensors = config['irreps_message_tensors'],

            # message batch normalization
            batch_normalize_msg = config['batch_normalize_msg'],

            # node update weights mlp
            node_update_hidden_layers = [config['node_update_hidden_layers']] if config['node_update_hidden_layers'] != 0 else [],
            node_update_act = acts[config['weights_act']],

            # geometric node format
            irreps_node_scalars = config['irreps_node_scalars'], 
            irreps_node_vectors = config['irreps_node_vectors'], 
            irreps_node_tensors = config['irreps_node_tensors'],

            # node update batch normalization
            batch_normalize_update=config['batch_normalize_update'],

            # interaction tp weights mlp
            basis_density_per_A = config['basis_density_per_A'],
            inter_tp_weights_hidden_layers = [config['inter_tp_weights_hidden_layers']] if config['inter_tp_weights_hidden_layers'] != 0 else [],
            inter_tp_weights_act = acts[config['weights_act']],

            # interaction tp spherical harmonics
            inter_spherical_harmonics_l = config['spherical_harmonics_l'],

            # general
            n_pattern_layers = config['n_pattern_layers'],
            radius = config['radius']
        )
    
    return inter_pred

def main(
        config,
        epochs,
        abs_path,
        data_dir,
        split_file,
        storage_path
):
    from prepare_pdbbind import pdb_ignore_list
    from prepare_pdbbind import defined_interactions

    df = pd.read_csv(abs_path + split_file, index_col=0)
    train_pdbs = [ x for x in list(df[(df['new_split'] == 'train') & df.CL1 & ~df.covalent].index) if x not in pdb_ignore_list ]
    test_pdbs = [ x for x in list(df[(df['new_split'] == 'test') & df.CL1 & ~df.covalent].index) if x not in pdb_ignore_list ]
    val_pdbs = [ x for x in list(df[(df['new_split'] == 'val') & df.CL1 & ~df.covalent].index) if x not in pdb_ignore_list ]

    validation_loss = 0.0

    for interaction_type in defined_interactions:
        print("@@@@ ", interaction_type)
        inter_pred = create_model(config=config)

        print("Model weights:", sum(p.numel() for p in inter_pred.parameters() if p.requires_grad))

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(inter_pred.parameters(), lr=1e-3, amsgrad=True)

        dataset_train = PDBBindInteractionDataset(abs_path + data_dir, train_pdbs, interaction_type)
        dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, collate_fn=dataset_train.collate_fn, pin_memory=True, num_workers=10)
        dataset_val = PDBBindInteractionDataset(abs_path + data_dir, val_pdbs, interaction_type)
        dataloader_val = DataLoader(dataset_val, batch_size=64, shuffle=True, collate_fn=dataset_val.collate_fn, pin_memory=True, num_workers=10)

        print("Training size:", len(dataset_train), "Validation size:", len(dataset_val))

        validation_loss += train(epochs, 50, dataloader_train, dataloader_val, inter_pred, loss_fn, optimizer, save_weights=False) ** 2
        print("\n-------------------------------------------------------------\n")

    validation_loss = validation_loss ** 0.5
    print(f"RESULT: {validation_loss:>8f} \n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Trains a Kernel to predict biomolecular interactions')
    parser.add_argument('--epochs', type=int, required=True)
    
    parser.add_argument('--node_emb_hidden_layers', type=int, required=True)
    parser.add_argument('--node_embedding_size', type=int, required=True)

    parser.add_argument('--msg_weights_hidden_layers', type=int, required=True)
    parser.add_argument('--weights_act', required=True)

    parser.add_argument('--spherical_harmonics_l', type=int, required=True)

    parser.add_argument('--irreps_message_scalars', type=int, required=True)
    parser.add_argument('--irreps_message_vectors', type=int, required=True)
    parser.add_argument('--irreps_message_tensors', type=int, required=True)

    parser.add_argument('--batch_normalize_msg', type=bool, required=True)

    parser.add_argument('--node_update_hidden_layers', type=int, required=True)

    parser.add_argument('--irreps_node_scalars', type=int, required=True)
    parser.add_argument('--irreps_node_vectors', type=int, required=True)
    parser.add_argument('--irreps_node_tensors', type=int, required=True)

    parser.add_argument('--batch_normalize_update', type=bool, required=True)

    parser.add_argument('--basis_density_per_A', type=int, required=True)
    parser.add_argument('--inter_tp_weights_hidden_layers', type=int, required=True)
    parser.add_argument('--radius', type=float, required=True)

    parser.add_argument('--n_pattern_layers', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)

    parser.add_argument('--abs_path', required=True,
                        help='Absolute path for all in and out files')
    parser.add_argument('--data_dir', required=True,
                        help='Path to pdbbind within the project path')
    parser.add_argument('--split_file', required=True,
                        help='Path to LP_PDBBind split file within the project path')
    parser.add_argument('--storage_path', required=True,
                        help='Path to directory for saving results within the project path')
    
    args = parser.parse_args()

    config = {
        "node_emb_hidden_layers": args.node_emb_hidden_layers,
        "node_embedding_size": args.node_embedding_size,

        "msg_weights_hidden_layers": args.msg_weights_hidden_layers,
        "weights_act": args.weights_act,

        "spherical_harmonics_l": args.spherical_harmonics_l,

        "irreps_message_scalars": args.irreps_message_scalars,
        "irreps_message_vectors": args.irreps_message_vectors,
        "irreps_message_tensors": args.irreps_message_tensors,

        "batch_normalize_msg": args.batch_normalize_msg,

        "node_update_hidden_layers": args.node_update_hidden_layers,

        "irreps_node_scalars": args.irreps_node_scalars,
        "irreps_node_vectors": args.irreps_node_vectors,
        "irreps_node_tensors": args.irreps_node_tensors,

        "batch_normalize_update": args.batch_normalize_update,

        "basis_density_per_A": args.basis_density_per_A,
        "inter_tp_weights_hidden_layers": args.inter_tp_weights_hidden_layers,
        "radius": args.radius,

        "n_pattern_layers": args.n_pattern_layers,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }

    main(
        config=config,
        epochs=args.epochs,
        abs_path=args.abs_path,
        data_dir=args.data_dir,
        split_file=args.split_file,
        storage_path=args.storage_path
    )