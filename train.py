import torch
import numpy as np
import pandas as pd

from torch import nn
from data import PDBBindInteractionDataset
from torch.utils.data import DataLoader
from models.InteractionPredictor import InteractionPredictor
from e3nn import o3

import warnings
warnings.filterwarnings("ignore")


standard_config = {
    "radius": 7.5,
        
    "basis_density_per_A": 5,

    "out_scalars": 1,
    "out_vectors": 0,
    "out_tensors": 0,

    "spherical_harmonics_l": 2,

    "node_embedding_scalars": 4,
    "node_embedding_vectors": 12,
    "node_embedding_tensors": 4,

    "interaction_tp_lig_weights_hidden_layers":[32],
    "interaction_tp_rec_weights_hidden_layers":[32],
    "interaction_tp_lig_weights_act": nn.Sigmoid(),
    "interaction_tp_rec_weights_act": nn.Sigmoid(),

    "n_pattern_layers": 3,

    "node_emb_hidden_layers": [8],
    "node_emb_act": nn.ReLU(),

    "batch_normalize_msg": False,
    "batch_normalize_node_upd": True,

    "msg_weights_hidden_layers": [8],
    "msg_weights_act": nn.Sigmoid(),

    "node_update_hidden_layers": [24],
    "node_update_act": nn.ReLU(),
}

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

def interaction_epoch(dataloader, interaction_model, loss_fn, optimizer, device, preload=False):
    # Set the model to training mode - important for batch normalization and dropout layers
    interaction_model.train()
    total_loss = 0
    confusion_matrix = torch.zeros( (2,2), dtype=torch.int )
    for multi_g in dataloader:
        if not preload:
            multi_g = multi_g.to(device)

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

def interaction_eval(dataloader, interaction_model, loss_fn, device, preload=False):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    interaction_model.eval()
    total_loss = 0
    confusion_matrix = torch.zeros( (2,2), dtype=torch.int )

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for multi_g in dataloader:
            if not preload:
                multi_g = multi_g.to(device)

            predicted_interactions, edges = interaction_model(multi_g)
            predicted_interactions = predicted_interactions.squeeze()

            true_interactions = edge_interactions(multi_g.y, edges, multi_g.pdb)

            total_loss += loss_fn(predicted_interactions, true_interactions.float()).item() if loss_fn != None else 0
            confusion_matrix += confusion_matrix_calc( predicted_interactions, true_interactions )

    return total_loss, confusion_matrix

def train(num_epochs, eval_every_n_epochs, dataloader_train, dataloader_eval, interaction_model, loss_fn, optimizer, device, save_weights=True, weights_path="", weights_suffix="", preload=False):
    model_weights_path = weights_path + "model_" + weights_suffix + ".weights"
    best_eval_loss = 99999999.9
    for e in range(1, num_epochs+1):
        epoch_loss, confusion_matrix = interaction_epoch(dataloader_train, interaction_model, loss_fn, optimizer, device, preload)
        tp = confusion_matrix[0,0].item()
        fn = confusion_matrix[1,0].item()
        fp = confusion_matrix[0,1].item()
        recall = 100*tp/(tp+fn)
        precision = 0.0
        if tp + fp != 0:
            precision = 100*tp / (tp+fp)
        print(f"train loss: {epoch_loss/len(dataloader_train):>7.10f}  recall: {recall:>6.2f}%  precision: {precision:>6.2f}%  [{e:>5d}/{num_epochs:>5d}]")
        if e % eval_every_n_epochs == 0 or e == num_epochs:
            eval_loss, eval_conf_matrix = interaction_eval(dataloader_eval, interaction_model, loss_fn, device, preload)
            tp = eval_conf_matrix[0,0].item()
            fn = eval_conf_matrix[1,0].item()
            fp = eval_conf_matrix[0,1].item()
            recall = 100*tp/(tp+fn)
            precision = 0.0
            if tp + fp != 0:
                precision = 100*tp / (tp+fp)
            print(f"eval set loss: {eval_loss/len(dataloader_eval):>7.10f}  recall: {recall:>6.2f}%  precision: {precision:>6.2f}%")
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                if save_weights:
                    torch.save(interaction_model.state_dict(), model_weights_path)

    return eval_loss/len(dataloader_eval), model_weights_path

def main(
        config,
        epochs,
        batch_size,
        lr,
        abs_path,
        data_dir,
        split_file,
        storage_path,
        num_workers,
        full_dataset_on_gpu,
        save_weights,
        interactions=None,
):
    from prepare_pdbbind import pdb_ignore_list
    from prepare_pdbbind import defined_interactions

    if interactions == None:
        interactions = defined_interactions

    df = pd.read_csv(abs_path + split_file, index_col=0)
    train_pdbs = [ x for x in list(df[(df['new_split'] == 'train') & df.CL1 & ~df.covalent].index) if x not in pdb_ignore_list ]
    test_pdbs = [ x for x in list(df[(df['new_split'] == 'test') & df.CL1 & ~df.covalent].index) if x not in pdb_ignore_list ]
    val_pdbs = [ x for x in list(df[(df['new_split'] == 'val') & df.CL1 & ~df.covalent].index) if x not in pdb_ignore_list ]

    validation_loss = 0.0
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device.")

    print("Interactions:", interactions)

    dataloader_train = {}
    dataloader_val = {}
    dataloader_test = {}
    for interaction_type in interactions:
        dataset_train = PDBBindInteractionDataset(abs_path + data_dir, train_pdbs, interaction_type)
        dataloader_train[interaction_type] = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=dataset_train.collate_fn, pin_memory=True, num_workers=num_workers)
        dataset_val = PDBBindInteractionDataset(abs_path + data_dir, val_pdbs, interaction_type)
        dataloader_val[interaction_type] = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, collate_fn=dataset_val.collate_fn, pin_memory=True, num_workers=num_workers)
        dataset_test = PDBBindInteractionDataset(abs_path + data_dir, test_pdbs, interaction_type)
        dataloader_test[interaction_type] = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, collate_fn=dataset_test.collate_fn, pin_memory=True, num_workers=num_workers)

    if(full_dataset_on_gpu):
        print("Moving the whole dataset to GPU...")
        for interaction_type in interactions:
            train_gpu = []
            val_gpu = []
            for mutligraph in dataloader_train[interaction_type]:
                train_gpu.append(mutligraph.to(device))
            for mutligraph in dataloader_val[interaction_type]:
                val_gpu.append(mutligraph.to(device))
            dataloader_train[interaction_type] = train_gpu
            dataloader_val[interaction_type] = val_gpu

    print("Training size:", len(dataset_train), "Validation size:", len(dataset_val), "Test size:", len(dataset_test))
    print("Batch size:", batch_size, "Learning rate:", lr)
    print("Model config:")
    print(config)

    for interaction_type in interactions:
        print("@@@@ ", interaction_type)
        inter_pred = InteractionPredictor(config)
        inter_pred.to(device)

        print("Model weights:", sum(p.numel() for p in inter_pred.parameters() if p.requires_grad))

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(inter_pred.parameters(), lr=lr, amsgrad=True)

        loss, model_weights_path = train(epochs, 1, dataloader_train[interaction_type], 
                                 dataloader_val[interaction_type], inter_pred, 
                                 loss_fn, optimizer, device, 
                                 save_weights=save_weights, weights_path=storage_path, weights_suffix=interaction_type,
                                 preload=full_dataset_on_gpu)
        validation_loss += loss ** 2
        
        if save_weights:
            inter_pred.load_state_dict(torch.load(model_weights_path))
            test_loss, conf_matrix = interaction_eval(dataloader_test[interaction_type], inter_pred, loss_fn, device)
            tp = conf_matrix[0,0].item()
            fn = conf_matrix[1,0].item()
            fp = conf_matrix[0,1].item()
            recall = 100*tp/(tp+fn)
            precision = 0.0
            if tp + fp != 0:
                precision = 100*tp / (tp+fp)
            print(f"Test set loss: {test_loss/len(dataloader_test[interaction_type]):>7.10f}  recall: {recall:>6.2f}%  precision: {precision:>6.2f}%")

        print("\n-------------------------------------------------------------\n")



    validation_loss = validation_loss ** 0.5
    print(f"RESULT: {validation_loss:>8f} \n")

if __name__ == "__main__":
    import argparse

    acts = {
        "ReLU" : nn.ReLU(), 
        "Sigmoid" : nn.Sigmoid(), 
        "Hardtanh" : nn.Hardtanh(),
        "None" : None
    }

    parser = argparse.ArgumentParser(description='Trains a Kernel to predict biomolecular interactions')
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    parser.add_argument('--full_dataset_on_gpu', action='store_true')
    parser.add_argument('--save_weights', action='store_true')
    
    # TODO Use standard config for all default values
    parser.add_argument('--radius', type=float, default=standard_config['radius'])
    parser.add_argument('--basis_density_per_A', type=int, default=5)

    parser.add_argument('--spherical_harmonics_l', type=int, default=2)

    parser.add_argument('--node_embedding_scalars', type=int, default=4)
    parser.add_argument('--node_embedding_vectors', type=int, default=12)
    parser.add_argument('--node_embedding_tensors', type=int, default=4)

    parser.add_argument('--interaction_tp_lig_weights_hidden_layers', type=int, default=32)
    parser.add_argument('--interaction_tp_rec_weights_hidden_layers', type=int, default=32)
    
    parser.add_argument('--weights_act', default="Sigmoid")

    parser.add_argument('--n_pattern_layers', type=int, default=3)

    parser.add_argument('--node_emb_hidden_layers', type=int, default=8)
    parser.add_argument('--node_update_hidden_layers', type=int, default=24)
    parser.add_argument('--node_act', default="ReLU")

    parser.add_argument('--batch_normalize_node_upd', type=int, default=1)
    parser.add_argument('--batch_normalize_msg', type=int, default=0)
    
    parser.add_argument('--msg_weights_hidden_layers', type=int, default=8)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--abs_path', required=True,
                        help='Absolute path for all in and out files')
    parser.add_argument('--data_dir', required=True,
                        help='Path to pdbbind within the project path')
    parser.add_argument('--split_file', required=True,
                        help='Path to LP_PDBBind split file within the project path')
    parser.add_argument('--storage_path', required=True,
                        help='Path to directory for saving results within the project path')
    
    parser.add_argument('--interactions', nargs='+', help='<Required> Set flag', required=True)

    
    args = parser.parse_args()

    config = {
        "radius": args.radius,
        
        "basis_density_per_A": args.basis_density_per_A,

        "out_scalars": 1,
        "out_vectors": 0,
        "out_tensors": 0,

        "spherical_harmonics_l": args.spherical_harmonics_l,

        "node_embedding_scalars": args.node_embedding_scalars,
        "node_embedding_vectors": args.node_embedding_vectors,
        "node_embedding_tensors": args.node_embedding_tensors,

        "interaction_tp_lig_weights_hidden_layers":[args.interaction_tp_lig_weights_hidden_layers] if args.interaction_tp_lig_weights_hidden_layers != 0 else [],
        "interaction_tp_rec_weights_hidden_layers":[args.interaction_tp_rec_weights_hidden_layers] if args.interaction_tp_rec_weights_hidden_layers != 0 else [],
        "interaction_tp_lig_weights_act": acts[args.weights_act],
        "interaction_tp_rec_weights_act": acts[args.weights_act],

        "n_pattern_layers": args.n_pattern_layers,

        "node_emb_hidden_layers": [args.node_emb_hidden_layers] if args.node_emb_hidden_layers != 0 else [],
        "node_emb_act": acts[args.node_act],

        "batch_normalize_msg": args.batch_normalize_msg != 0,
        "batch_normalize_node_upd": args.batch_normalize_node_upd != 0,

        "msg_weights_hidden_layers": [args.msg_weights_hidden_layers] if args.msg_weights_hidden_layers != 0 else [],
        "msg_weights_act": acts[args.weights_act],

        "node_update_hidden_layers": [args.node_update_hidden_layers] if args.node_update_hidden_layers != 0 else [],
        "node_update_act": acts[args.node_act],
    }

    main(
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        abs_path=args.abs_path,
        data_dir=args.data_dir,
        split_file=args.split_file,
        storage_path=args.storage_path,
        num_workers=args.num_workers,
        full_dataset_on_gpu=args.full_dataset_on_gpu,
        interactions=args.interactions,
        save_weights=args.save_weights
    )