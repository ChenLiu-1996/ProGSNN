import datetime
import os
import sys
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import yaml
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.append('../')
from datasets.de_shaw_Dataset import DEShaw
from models.gsae import GSAE
from models.progsnn import ProGSNN
from torch_geometric.loader import DataLoader
from utils.attribute_hashmap import AttributeHashmap
from utils.parse import parse_hparams

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--config', help='Path to config yaml file.')
    parser.add_argument('--protein_name', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--save_root', default='../../train_logs/', type=str)
    parser.add_argument(
        '--dataset_path', default='../../input_graphs/', type=str)

    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)
    # parse params
    args = parser.parse_args()

    # Reproducibility.
    # According to https://github.com/pyg-team/pytorch_geometric/issues/92#issuecomment-472332656:
    # "However, on GPU, we can not guarantee determinism because we make heavy use of
    # scatter ops to implement Graph Neural Networks which are non-deterministic by nature on GPU."
    #
    # We cannot use `args.deterministic = True` because it's currently incompatible with the
    # scattering operation: `torch_scatter.scatter()`.
    # It will trigger the error:
    #     scatter_add_cuda_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)
    # According to https://github.com/pytorch/pytorch/issues/50469, this is a known imperfection.
    # We can only accept it that the randoms state control is not in place for the model training.
    pl.seed_everything(args.random_seed)

    # Load the model hyperparams from the config file.
    default_config_loc = {
        'gsae': '../../config/gsae.yaml',
        'progsnn': '../../config/progsnn.yaml',
    }
    if args.config is None:
        args.config = default_config_loc[args.model]

    model_hparams = AttributeHashmap(yaml.safe_load(open(args.config)))
    model_hparams = parse_hparams(model_hparams)
    model_hparams.alpha = args.alpha

    full_dataset = DEShaw('%s/%s/total_graphs.pkl' %
                          (args.dataset_path, args.protein_name))

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset,
                                                       [train_size, val_size],
                                                       generator=torch.Generator().manual_seed(args.random_seed))

    # train loader
    train_loader = DataLoader(train_set,
                              batch_size=model_hparams.batch_size,
                              shuffle=True,
                              num_workers=model_hparams.num_workers)
    # validation loader
    val_loader = DataLoader(val_set,
                            batch_size=model_hparams.batch_size,
                            shuffle=False,
                            num_workers=model_hparams.num_workers)

    # logger
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%M")
    save_dir = '%s/%s/%s_logs_run_deshaw_%s_alpha=%s_%s/' % (
        args.save_root, args.protein_name, args.model, args.protein_name, args.alpha, date_suffix)

    # early stopping
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0.0,
                                        patience=model_hparams.patience,
                                        verbose=False,
                                        mode='min')

    # init module
    model_hparams.input_dim = train_set[0].x.shape[-1]
    if args.model == 'gsae':
        model_hparams.len_epoch = len(train_loader)
        model = GSAE(hparams=model_hparams)
    elif args.model == 'progsnn':
        model_hparams.prot_graph_size = max(
            [item.edge_index.shape[1] for item in full_dataset])
        model = ProGSNN(hparams=model_hparams)

    # most basic trainer, uses good defaults
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=model_hparams.n_epochs,
        accelerator='gpu',
        devices=model_hparams.gpu_ids,
        gradient_clip_val=model_hparams.grad_clip,
        callbacks=[early_stop_callback],
        logger=False,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    model = model.cpu()
    model.dev_type = 'cpu'

    with torch.no_grad():
        loss = model.get_loss_list()

    loss = np.array(loss)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir + "reg_loss_list.npy", loss)

    print('saving model to %s' % save_dir)
    torch.save(model.state_dict(), save_dir + 'model.pt')
