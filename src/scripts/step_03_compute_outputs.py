import os
import sys
from argparse import ArgumentParser
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.append('../')
from datasets.de_shaw_Dataset import DEShaw
from models.gsae import GSAE
from models.progsnn import ProGSNN
from utils.attribute_hashmap import AttributeHashmap
from utils.parse import parse_hparams

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', help='Path to config yaml file.')
    parser.add_argument('--protein_name', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--save_root', default='../../train_logs/', type=str)
    parser.add_argument(
        '--dataset_path', default='../../input_graphs/', type=str)
    parser.add_argument(
        '--output_root', default='../../output_numpy/', type=str)

    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)
    # parse params
    args = parser.parse_args()

    # Reproducibility.
    pl.seed_everything(args.random_seed)
    args.deterministic = True

    save_dir = glob('%s/%s/%s_logs_run_deshaw_%s_alpha=%s*/' % (args.save_root,
                    args.protein_name, args.model, args.protein_name, args.alpha))

    print('A total of %s model weights found.' % len(save_dir))
    if len(save_dir) > 0:
        save_dir = sorted(save_dir)[-1]

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

    full_loader = DataLoader(full_dataset,
                             batch_size=model_hparams.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)

    device = 'cuda' if len(model_hparams.gpu_ids) > 0 else 'cpu'
    model_hparams.input_dim = full_dataset[0].x.shape[-1]
    if args.model == 'gsae':
        model_hparams.len_epoch = len(full_loader)
        model = GSAE(hparams=model_hparams).to(device)
    elif args.model == 'progsnn':
        model_hparams.prot_graph_size = max(
            [item.edge_index.shape[1] for item in full_dataset])
        model = ProGSNN(hparams=model_hparams).to(device)

    model.load_state_dict(torch.load(save_dir + 'model.pt'))
    model.eval()

    embeddings, times = [], []
    with torch.no_grad():
        for batch in tqdm(full_loader):
            batch = batch.to(device)
            if args.model == 'gsae':
                z, mu, logvar, coeffs = model.encode(batch)
                embed = mu
            elif args.model == 'progsnn':
                z_rep, coeffs, coeffs_recon = model.encode(batch)
                embed = z_rep
            embeddings.append(embed.cpu().numpy())
            times.append(batch.time.cpu().numpy())

    times = np.hstack(times)
    embeddings = np.vstack(embeddings)

    output_dir = '%s/%s/%s_%s_alpha=%s/' % (args.output_root, args.protein_name,
                                            args.model, args.protein_name, args.alpha)

    os.makedirs(output_dir, exist_ok=True)
    np.save(output_dir + 'embeddings.npy', embeddings)
    np.save(output_dir + 'times.npy', times)

    os.system('rm -rf ./checkpoints/')
