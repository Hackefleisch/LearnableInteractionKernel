import torch
import pandas as pd

from torch import nn
from data import PDBBindInteractionDataset
from torch.utils.data import DataLoader
from model import InteractionPredictorPointwise
from train import train, interaction_eval

import warnings
warnings.filterwarnings("ignore")

def main(
        config,
        epochs,
        batch_size,
        abs_path,
        data_dir,
        split_file,
        storage_path,
        num_workers,
        full_dataset_on_gpu,
):
    from prepare_pdbbind import pdb_ignore_list
    from prepare_pdbbind import defined_interactions

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

    dataloader_train = {}
    dataloader_val = {}
    dataloader_test = {}
    for interaction_type in defined_interactions:
        dataset_train = PDBBindInteractionDataset(abs_path + data_dir, train_pdbs, interaction_type)
        dataloader_train[interaction_type] = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=dataset_train.collate_fn, pin_memory=True, num_workers=num_workers)
        dataset_val = PDBBindInteractionDataset(abs_path + data_dir, val_pdbs, interaction_type)
        dataloader_val[interaction_type] = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, collate_fn=dataset_val.collate_fn, pin_memory=True, num_workers=num_workers)
        dataset_test = PDBBindInteractionDataset(abs_path + data_dir, test_pdbs, interaction_type)
        dataloader_test[interaction_type] = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, collate_fn=dataset_val.collate_fn, pin_memory=True, num_workers=num_workers)

    if(full_dataset_on_gpu):
        print("Moving the whole dataset to GPU...")
        for interaction_type in defined_interactions:
            train_gpu = []
            val_gpu = []
            for mutligraph in dataloader_train[interaction_type]:
                train_gpu.append(mutligraph.to(device))
            for mutligraph in dataloader_val[interaction_type]:
                val_gpu.append(mutligraph.to(device))
            dataloader_train[interaction_type] = train_gpu
            dataloader_val[interaction_type] = val_gpu

    print("Training size:", len(dataset_train), "Validation size:", len(dataset_val), "Test size:", len(dataset_test), '\n')

    for interaction_type in defined_interactions:
        print("@@@@ ", interaction_type)
        inter_pred = InteractionPredictorPointwise(config=config)
        inter_pred.to(device)

        print("Model weights:", sum(p.numel() for p in inter_pred.parameters() if p.requires_grad))

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(inter_pred.parameters(), lr=config["lr"], amsgrad=True)

        validation_loss += train(epochs, 50, dataloader_train[interaction_type], dataloader_val[interaction_type], inter_pred, loss_fn, optimizer, device, save_weights=False, preload=full_dataset_on_gpu) ** 2
        test_loss, test_conf_matrix = interaction_eval(dataloader_test[interaction_type], inter_pred, loss_fn, device)
        tp = test_conf_matrix[0,0].item()
        fn = test_conf_matrix[1,0].item()
        fp = test_conf_matrix[0,1].item()
        recall = 100*tp/(tp+fn)
        precision = 0.0
        if tp + fp != 0:
            precision = 100*tp / (tp+fp)
        print(f"@@@@  test set loss: {test_loss/len(dataloader_test[interaction_type]):>7.10f}  recall: {recall:>6.2f}%  precision: {precision:>6.2f}%")
        print("\n-------------------------------------------------------------\n")

    validation_loss = validation_loss ** 0.5
    print(f"RESULT: {validation_loss:>8f} \n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Trains a Kernel to predict biomolecular interactions')
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    parser.add_argument('--full_dataset_on_gpu', action='store_true')
    
    parser.add_argument('--node_embedding_size', type=int, required=True)
    parser.add_argument('--conv_radius', type=float, required=True)
    parser.add_argument('--node_emb_hidden_layers', type=int, required=True)
    parser.add_argument('--basis_density_per_A', type=int, required=True)

    parser.add_argument('--spherical_harmonics_l', type=int, required=True)

    parser.add_argument('--irreps_node_scalars', type=int, required=True)
    parser.add_argument('--irreps_node_vectors', type=int, required=True)
    parser.add_argument('--irreps_node_tensors', type=int, required=True)

    parser.add_argument('--tp_weights_hidden_layers', type=int, required=True)
    parser.add_argument('--tp_weights_act', required=True)

    parser.add_argument('--batch_normalize', type=int, required=True)

    parser.add_argument('--radius', type=float, required=True)
    parser.add_argument('--inter_tp_weights_hidden_layers', type=int, required=True)
    parser.add_argument('--inter_tp_weights_act', required=True)

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

    acts = {
        "ReLU" : nn.ReLU(), 
        "Sigmoid" : nn.Sigmoid(), 
        "Hardtanh" : nn.Hardtanh()
    }

    config = {
        "node_emb_hidden_layers": [args.node_emb_hidden_layers] if args.node_emb_hidden_layers != 0 else [],
        "node_embedding_size": args.node_embedding_size,
        "conv_radius": args.conv_radius,
        "basis_density_per_A": args.basis_density_per_A,

        "spherical_harmonics_l": args.spherical_harmonics_l,

        "irreps_node_scalars": args.irreps_node_scalars,
        "irreps_node_vectors": args.irreps_node_vectors,
        "irreps_node_tensors": args.irreps_node_tensors,

        "tp_weights_hidden_layers": [args.tp_weights_hidden_layers] if args.tp_weights_hidden_layers != 0 else [],
        "tp_weights_act": acts[args.tp_weights_act],

        "batch_normalize": args.batch_normalize,

        "radius": args.radius,
        "inter_tp_weights_hidden_layers": [args.inter_tp_weights_hidden_layers] if args.inter_tp_weights_hidden_layers != 0 else [],
        "inter_tp_weights_act": acts[args.inter_tp_weights_act],

        "batch_size": args.batch_size,
        "lr": args.lr,
    }

    main(
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        abs_path=args.abs_path,
        data_dir=args.data_dir,
        split_file=args.split_file,
        storage_path=args.storage_path,
        num_workers=args.num_workers,
        full_dataset_on_gpu=args.full_dataset_on_gpu
    )