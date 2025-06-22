import logging
from util import set_seed, logger_setup
from data_loading import get_data

# copied from inference.py
import torch
from torch import Tensor
from torch.nn import Parameter
import pandas as pd
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo
from training import get_model
from torch_geometric.nn import summary
import wandb
import os
import sys
import time

# copied from train_util.py
import tqdm
from torch_geometric.transforms import BaseTransform
from typing import Union
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import f1_score
import json

# Torch-related library
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer, ModelConfig, ExplainerConfig
from torch_geometric.explain.config import MaskType, ModelMode
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks




def loading_helper(loader, inds, model, data, device, args):
    '''Code is based on evaluate_homo function in train_util.py'''
    counter = 0
    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        counter += 1

        # select the seed edges from which the batch was created
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        # add the seed edges that have not been sampled to the batch
        missing = ~torch.isin(batch_edge_ids, batch.edge_attr[:, 0].detach().cpu())

        if missing.sum() != 0 and (args.data == 'Small_J' or args.data == 'Small_Q'):
            missing_ids = batch_edge_ids[missing].int()
            n_ids = batch.n_id
            add_edge_index = data.edge_index[:, missing_ids].detach().clone()
            node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
            add_edge_index = torch.tensor([[node_mapping[val.item()] for val in row] for row in add_edge_index])
            add_edge_attr = data.edge_attr[missing_ids, :].detach().clone()
            add_y = data.y[missing_ids].detach().clone()

            batch.edge_index = torch.cat((batch.edge_index, add_edge_index), 1)
            batch.edge_attr = torch.cat((batch.edge_attr, add_edge_attr), 0)
            batch.y = torch.cat((batch.y, add_y), 0)

            mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))

        # remove the unique edge id from the edge features, as it's no longer needed
        batch.edge_attr = batch.edge_attr[:, 1:]

        with torch.no_grad():
            batch.to(device)

    print(f'number of batch is {counter}')
    return loader, inds, model, data, device, args, batch_edge_ids, batch_edge_inds, batch


def load_trained_model_homo(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    """
    Code is based on inference.py
    """
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define a model config dictionary and wandb logging at the same time
    wandb.init(
        mode="disabled" if args.testing else "online",
        project="explainability",

        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
        }
    )

    config = wandb.config

    # set the transform if ego ids should be used
    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    # add the unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform,
                                                   args)

    # get the model
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)

    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')


    logging.info("=> loading model checkpoint")
    # todo: to avoid issue: hardcoding unique name as directed

    checkpoint = torch.load(f'{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    logging.info("=> loaded checkpoint (epoch {})".format(start_epoch))



    if not args.reverse_mp:
        te_loader, te_inds, model, te_data, device, args, te_batch_edge_ids, te_batch_edge_inds, te_batch = loading_helper(
            te_loader, te_inds, model, te_data, device, args)
        val_loader, val_inds, _, val_data, _, _, val_batch_edge_ids, val_batch_edge_inds, val_batch = loading_helper(val_loader, val_inds, model, val_data, device, args)
    else:
        #todo: update handling and commenting related to block normal GINe with reverse_mp.
        raise ValueError("Graph cannot be Heterograph.")

    parts_dict = {"te_loader": te_loader,
             "te_inds": te_inds,
             "te_data": te_data,
             "te_batch_edge_ids": te_batch_edge_ids,
             "te_batch_edge_inds": te_batch_edge_inds,
             "te_batch": te_batch,
             "val_loader": val_loader,
             "val_inds": val_inds,
             "val_data": val_data,
             "val_batch_edge_ids": val_batch_edge_ids,
             "val_batch_edge_inds": val_batch_edge_inds,
             "val_batch": val_batch,
             "model": model,
             "device": device,
             "args": args,
    }
    wandb.finish()

    return parts_dict


def reload_main(args):
    with open('data_config.json', 'r') as config_file:
        data_config = json.load(config_file)

    # Setup logging
    logger_setup()


    # set seed
    set_seed(args.seed)

    # get data
    logging.info("Retrieving data")
    t1 = time.perf_counter()

    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)

    t2 = time.perf_counter()
    logging.info(f"Retrieved data in {t2 - t1:.2f}s")

    logging.info(f"Running Explanation")

    parts_dict = load_trained_model_homo(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)

    return parts_dict


