
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


def start_points():
    params1 = {
        "node_emb_hidden_layers": [],
        "node_embedding_size": 8,

        "msg_weights_hidden_layers": [24],
        "weights_act": "ReLU",

        "spherical_harmonics_l": 2,

        "irreps_message_scalars": 8,
        "irreps_message_vectors": 1,
        "irreps_message_tensors": 1,

        "batch_normalize_msg": True,

        "node_update_hidden_layers": [24],

        "irreps_node_scalars": 8,
        "irreps_node_vectors": 1,
        "irreps_node_tensors": 1,

        "batch_normalize_update": False,

        "basis_density_per_A": 5,
        "inter_tp_weights_hidden_layers": [24],
        "radius": 7.5,

        "n_pattern_layers": 2,
        "batch_size": 128,
        "lr": 1e-3,
    }
    params2 = {
        "node_emb_hidden_layers": [],
        "node_embedding_size": 8,

        "msg_weights_hidden_layers": [24],
        "weights_act": "ReLU",

        "spherical_harmonics_l": 2,

        "irreps_message_scalars": 8,
        "irreps_message_vectors": 1,
        "irreps_message_tensors": 1,

        "batch_normalize_msg": True,

        "node_update_hidden_layers": [24],

        "irreps_node_scalars": 8,
        "irreps_node_vectors": 1,
        "irreps_node_tensors": 1,

        "batch_normalize_update": False,

        "basis_density_per_A": 5,
        "inter_tp_weights_hidden_layers": [24],
        "radius": 7.5,

        "n_pattern_layers": 3,  # <---- The only change to params1
        "batch_size": 128,
        "lr": 1e-3,
    }
    return [params1, params2]

def create_model(config):
    acts = {
        "ReLU" : nn.ReLU(), 
        "Sigmoid" : nn.Sigmoid(), 
        "Hardtanh" : nn.Hardtanh()
    }

    inter_pred = InteractionPredictor(
            # node embedding mlp
            node_emb_hidden_layers = config['node_emb_hidden_layers'],
            node_embedding_size = config['node_embedding_size'],

            # message weights mlp
            msg_weights_hidden_layers = config['msg_weights_hidden_layers'],
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
            node_update_hidden_layers = config['node_update_hidden_layers'],
            node_update_act = acts[config['weights_act']],

            # geometric node format
            irreps_node_scalars = config['irreps_node_scalars'], 
            irreps_node_vectors = config['irreps_node_vectors'], 
            irreps_node_tensors = config['irreps_node_tensors'],

            # node update batch normalization
            batch_normalize_update=config['batch_normalize_update'],

            # interaction tp weights mlp
            basis_density_per_A = config['basis_density_per_A'],
            inter_tp_weights_hidden_layers = config['inter_tp_weights_hidden_layers'],
            inter_tp_weights_act = acts[config['weights_act']],

            # interaction tp spherical harmonics
            inter_spherical_harmonics_l = config['spherical_harmonics_l'],

            # general
            n_pattern_layers = config['n_pattern_layers'],
            radius = config['radius']
        )
    
    return inter_pred

def train_interaction(config, data_dir="pdbbind2020/", split_file='LP_PDBBind.csv', interaction_types=[], max_epochs=1000, num_workers=8):
    models = {}
    optimizers = {}
    loss_fns = {}
    for interaction_type in interaction_types:
        models[interaction_type] = create_model(config)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                models[interaction_type] = nn.DataParallel(models[interaction_type])
        models[interaction_type].to(device)

        optimizers[interaction_type] = torch.optim.Adam(models[interaction_type].parameters(), lr=config['lr'], amsgrad=True)
        loss_fns[interaction_type] = nn.BCEWithLogitsLoss()

    num_weights = sum(p.numel() for p in models[interaction_types[0]].parameters() if p.requires_grad)
    print("Model weights:", num_weights)

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            for interaction_type in interaction_types:
                models[interaction_type].load_state_dict(checkpoint_state["model_state_dicts"][interaction_type])
                optimizers[interaction_type].load_state_dict(checkpoint_state["optimizer_state_dicts"][interaction_type])
    else:
        start_epoch = 1

    trainloaders = {}
    valloaders = {}

    for interaction_type in interaction_types:
        trainset, valset, _ = load_data(data_dir, split_file, interaction_type)

        trainloaders[interaction_type] = torch.utils.data.DataLoader(
            trainset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=num_workers, collate_fn=trainset.collate_fn
        )
        valloaders[interaction_type] = torch.utils.data.DataLoader(
            valset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=num_workers, collate_fn=valset.collate_fn
        )

    for epoch in range(start_epoch, max_epochs+1):  # loop over all datasets multiple times
        model_state_dicts = {}
        optimizer_state_dicts = {}

        eval_losses = {}

        for interaction_type in interaction_types:
            _, _ = interaction_epoch(trainloaders[interaction_type], models[interaction_type], loss_fns[interaction_type], optimizers[interaction_type], device)

            eval_losses[interaction_type], _ = interaction_eval(valloaders[interaction_type], models[interaction_type], loss_fns[interaction_type], device)
            eval_losses[interaction_type] /= len(valloaders[interaction_type])

            model_state_dicts[interaction_type] = models[interaction_type].state_dict()
            optimizer_state_dicts[interaction_type] = optimizers[interaction_type].state_dict()

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dicts": model_state_dicts,
            "optimizer_state_dicts": optimizer_state_dicts,
        }
        report = {}
        for interaction_type in interaction_types:
            report[interaction_type + "_loss"] = eval_losses[interaction_type]
        report['weights'] = num_weights
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                report,
                checkpoint=checkpoint,
            )

def test_evaluation(inter_pred, device="cpu", data_dir="pdbbind2020/", split_file='LP_PDBBind.csv', interaction_types=[], batch_size=64, num_workers=8):
    precisions = {}
    recalls = {}
    for interaction_type in interaction_types:
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

        precisions[interaction_type] = precision
        recalls[interaction_type] = recall

    return recalls, precisions


def main(num_samples=4, num_epochs=10, gpus_per_trial=1, cpus_per_trial=4,
         project_path_absolute='/home/iwe20/Projects/LearnableInteractionKernel/',
         data_dir="pdbbind2020/",
         split_file='LP_PDBBind.csv',
         storage_path='hyperparam_tuning/'
        ):
    from prepare_pdbbind import defined_interactions

    config = {
        "node_emb_hidden_layers": tune.choice([[], [8]]),
        "node_embedding_size": tune.choice([4, 8, 16]),

        "msg_weights_hidden_layers": tune.choice([[], [8], [16], [24]]),
        "weights_act": tune.choice(["ReLU", "Sigmoid", "Hardtanh"]),

        "spherical_harmonics_l": tune.choice([1, 2]),

        "irreps_message_scalars": tune.choice([4, 8, 16]),
        "irreps_message_vectors": tune.choice([1, 3, 6]),
        "irreps_message_tensors": tune.choice([0, 1, 2]),

        "batch_normalize_msg": tune.choice([True, False]),

        "node_update_hidden_layers": tune.choice([[], [8], [16], [24]]),

        "irreps_node_scalars": tune.choice([4, 8, 16]),
        "irreps_node_vectors": tune.choice([1, 3, 6]),
        "irreps_node_tensors": tune.choice([0, 1, 2]),

        "batch_normalize_update": tune.choice([True, False]),

        "basis_density_per_A": tune.choice([1, 5, 10]),
        "inter_tp_weights_hidden_layers": tune.choice([[], [8], [16], [24]]),
        "radius": tune.choice([7.5]),

        "n_pattern_layers": tune.choice([1, 2, 3, 4]),
        "batch_size": tune.choice([64, 128, 256]),
        "lr": tune.choice([1e-2, 1e-3, 1e-4]),
    }

    metrics = [interaction_type + "_loss" for interaction_type in defined_interactions]


    searcher = OptunaSearch(
        space=config,
        metric=metrics,
        mode=["min" for _ in defined_interactions],
        points_to_evaluate=start_points()
    )

    result = tune.run(
        partial(train_interaction, data_dir=project_path_absolute + data_dir, split_file=project_path_absolute + split_file, interaction_types=defined_interactions, max_epochs=num_epochs, num_workers=cpus_per_trial),
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        num_samples=num_samples,
        search_alg=searcher,
        raise_on_failed_trial=False,
        name="InteractionPrediction",
        storage_path=project_path_absolute + storage_path
    )

    best_trial = result.get_best_trial("hbond_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation hbond_loss: {best_trial.last_result['hbond_loss']}")
    print(f"Best trial final validation hydrophobic_loss: {best_trial.last_result['hydrophobic_loss']}")
    print(f"Best trial final validation pistacking_loss: {best_trial.last_result['pistacking_loss']}")
    print(f"Best trial final validation halogenbond_loss: {best_trial.last_result['halogenbond_loss']}")
    print(f"Best trial final validation saltbridges_loss: {best_trial.last_result['saltbridges_loss']}")
    print(f"Best trial final validation pication_loss: {best_trial.last_result['pication_loss']}")

    best_trained_model = create_model(best_trial.config)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="hbond_loss", mode="min")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["model_state_dicts"]["hbond"])
        test_recalls, test_precisions = test_evaluation(best_trained_model, device, data_dir=project_path_absolute + data_dir, split_file=project_path_absolute + split_file, interaction_types=defined_interactions, batch_size=64, num_workers=8)
        for interaction_type in defined_interactions:
            print("Best trial test set", interaction_type, "recall: {} precision: {}".format(test_recalls[interaction_type], test_precisions[interaction_type]))

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=1, num_epochs=5, gpus_per_trial=1, cpus_per_trial=12)