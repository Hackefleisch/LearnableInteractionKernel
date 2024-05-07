
import torch
from torch import nn

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from ray.tune.search.optuna import OptunaSearch

import pandas as pd
import tempfile
from pathlib import Path
from functools import partial

from prepare_pdbbind import pdb_ignore_list
from data import PDBBindInteractionDataset
from model import InteractionPredictor

from train import edge_interactions
from train import interaction_epoch
from train import interaction_eval

def load_data(data_dir, split_file, interaction_type):
    df = pd.read_csv(split_file, index_col=0)
    train_pdbs=[ x for x in list(df[(df['new_split'] == 'train') & df.CL1 & ~df.covalent].index) if x not in pdb_ignore_list ]
    val_pdbs=[ x for x in list(df[(df['new_split'] == 'val') & df.CL1 & ~df.covalent].index) if x not in pdb_ignore_list ]
    test_pdbs=[ x for x in list(df[(df['new_split'] == 'test') & df.CL1 & ~df.covalent].index) if x not in pdb_ignore_list ]
    dataset_train = PDBBindInteractionDataset(data_dir, train_pdbs, interaction_type)
    dataset_val = PDBBindInteractionDataset(data_dir, val_pdbs, interaction_type)
    dataset_test = PDBBindInteractionDataset(data_dir, test_pdbs, interaction_type)

    return dataset_train, dataset_val, dataset_test

def create_model(config):
    inter_pred = InteractionPredictor(
            # node embedding mlp
            node_emb_hidden_layers = config['node_emb_hidden_layers'],
            node_embedding_size = config['node_embedding_size'],

            # message weights mlp
            msg_weights_hidden_layers = config['msg_weights_hidden_layers'],
            msg_weights_act = config['msg_weights_act'],

            # message tp spherical harmonics edge
            pattern_spherical_harmonics_l = config['pattern_spherical_harmonics_l'],

            # message format
            irreps_message_scalars = config['irreps_message_scalars'], 
            irreps_message_vectors = config['irreps_message_vectors'], 
            irreps_message_tensors = config['irreps_message_tensors'],

            # message batch normalization
            batch_normalize_msg = config['batch_normalize_msg'],

            # node update weights mlp
            node_update_hidden_layers = config['node_update_hidden_layers'],
            node_update_act = config['node_update_act'],

            # geometric node format
            irreps_node_scalars = config['irreps_node_scalars'], 
            irreps_node_vectors = config['irreps_node_vectors'], 
            irreps_node_tensors = config['irreps_node_tensors'],

            # node update batch normalization
            batch_normalize_update=config['batch_normalize_update'],

            # interaction tp weights mlp
            basis_density_per_A = config['basis_density_per_A'],
            inter_tp_weights_hidden_layers = config['inter_tp_weights_hidden_layers'],
            inter_tp_weights_act = config['inter_tp_weights_act'],

            # interaction tp spherical harmonics
            inter_spherical_harmonics_l = config['inter_spherical_harmonics_l'],

            # general
            n_pattern_layers = config['n_pattern_layers'],
            radius = config['radius']
        )
    
    return inter_pred

def train_interaction(config, data_dir="pdbbind2020/", split_file='LP_PDBBind.csv', interaction_type="", max_epochs=1000, num_workers=8):
    inter_pred = create_model(config)

    num_weights = sum(p.numel() for p in inter_pred.parameters() if p.requires_grad)
    print("Model weights:", num_weights)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    inter_pred.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(inter_pred.parameters(), lr=config['lr'], amsgrad=True)

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 1

    trainset, valset, _ = load_data(data_dir, split_file, interaction_type)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=num_workers, collate_fn=trainset.collate_fn
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=num_workers, collate_fn=valset.collate_fn
    )


    for epoch in range(start_epoch, max_epochs+1):  # loop over the dataset multiple times
        epoch_loss, confusion_matrix = interaction_epoch(trainloader, inter_pred, loss_fn, optimizer, device)
        tp = confusion_matrix[0,0].item()
        fn = confusion_matrix[1,0].item()
        fp = confusion_matrix[0,1].item()
        recall = tp/(tp+fn)
        precision = 0.0
        if tp + fp != 0:
            precision = tp / (tp+fp)

        #print(f"weights: {num_weights} loss: {epoch_loss/len(trainloader):>7.10f}  recall: {100*recall:>6.2f}%  precision: {100*precision:>6.2f}%  [{epoch:>5d}/{max_epochs:>5d}]")


        eval_loss, eval_conf_matrix = interaction_eval(valloader, inter_pred, loss_fn, device)
        tp = eval_conf_matrix[0,0].item()
        fn = eval_conf_matrix[1,0].item()
        fp = eval_conf_matrix[0,1].item()
        recall = tp/(tp+fn)
        precision = 0.0
        if tp + fp != 0:
            precision = tp / (tp+fp)

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": inter_pred.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"loss": eval_loss / len(valloader), "precision": precision, "recall": recall, 'interaction' : interaction_type, 'weights' : num_weights},
                checkpoint=checkpoint,
            )

    print("Finished Training")

def test_evaluation(inter_pred, device="cpu", data_dir="pdbbind2020/", split_file='LP_PDBBind.csv', interaction_type="", batch_size=64, num_workers=8):
    _, _, testset = load_data(data_dir, split_file, interaction_type)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=testset.collate_fn
    )

    _, eval_conf_matrix = interaction_eval(testloader, inter_pred, None, device)

    tp = eval_conf_matrix[0,0]
    fn = eval_conf_matrix[1,0]
    fp = eval_conf_matrix[0,1]
    recall = tp/(tp+fn)
    precision = 0.0
    if tp + fp != 0:
        precision = tp / (tp+fp)

    return recall, precision


def main(num_samples=4, min_num_epochs=1, max_num_epochs=10, gpus_per_trial=1, cpus_per_trial=4):
    config = {
        "node_emb_hidden_layers": tune.choice([[], [4], [16]]),
        "node_embedding_size": tune.choice([4, 8, 16, 32]),

        "msg_weights_hidden_layers": tune.choice([[], [8], [16], [32]]),
        "msg_weights_act": tune.choice([nn.ReLU(), nn.Sigmoid(), nn.Hardtanh()]),

        "pattern_spherical_harmonics_l": tune.choice([0, 1, 2]),

        "irreps_message_scalars": tune.choice([4, 8, 16, 32]),
        "irreps_message_vectors": tune.choice([0, 1, 3, 6]),
        "irreps_message_tensors": tune.choice([0, 1, 2]),

        "batch_normalize_msg": tune.choice([True, False]),

        "node_update_hidden_layers": tune.choice([[], [8], [16], [32]]),
        "node_update_act": tune.choice([nn.ReLU(), nn.Sigmoid(), nn.Hardtanh()]),

        "irreps_node_scalars": tune.choice([4, 8, 16, 32]),
        "irreps_node_vectors": tune.choice([0, 1, 3, 6]),
        "irreps_node_tensors": tune.choice([0, 1, 2]),

        "batch_normalize_update": tune.choice([True, False]),

        "basis_density_per_A": tune.choice([1, 5, 10, 20]),
        "inter_tp_weights_hidden_layers": tune.choice([[], [8], [16], [32]]),
        "inter_tp_weights_act": tune.choice([nn.ReLU(), nn.Sigmoid(), nn.Hardtanh()]),
        "radius": tune.choice([7.5]),

        "inter_spherical_harmonics_l": tune.choice([0, 1, 2]),

        "n_pattern_layers": tune.choice([1, 2, 3, 4]),
        "batch_size": tune.choice([16, 32, 64, 128, 256]),
        "lr": tune.choice([1e-2, 1e-3, 1e-4]),
    }

    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric="recall",
        mode="max",
        max_t=max_num_epochs,
        grace_period=min_num_epochs,
        reduction_factor=3,
    )

    searcher = OptunaSearch(
        space=config,
        metric="recall",
        mode="max"
    )

    data_dir="/home/iwe20/Projects/LearnableInteractionKernel/pdbbind2020/"
    split_file='/home/iwe20/Projects/LearnableInteractionKernel/LP_PDBBind.csv'
    interaction_type="hydrophobic"

    result = tune.run(
        partial(train_interaction, data_dir=data_dir, split_file=split_file, interaction_type=interaction_type, max_epochs=max_num_epochs, num_workers=cpus_per_trial),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        #config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=searcher,
        raise_on_failed_trial=False
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation recall: {best_trial.last_result['recall']}")
    print(f"Best trial final validation precision: {best_trial.last_result['precision']}")

    best_trained_model = create_model(best_trial.config)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="recall", mode="max")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        test_recall, test_precision = test_evaluation(best_trained_model, device, data_dir=data_dir, split_file=split_file, interaction_type=interaction_type, batch_size=64, num_workers=8)
        print("Best trial test set recall: {} precision: {}".format(test_recall, test_precision))

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=15, min_num_epochs=1, max_num_epochs=40, gpus_per_trial=1, cpus_per_trial=4)