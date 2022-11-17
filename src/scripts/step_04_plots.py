import os
from argparse import ArgumentParser

import numpy as np
import phate
import pytorch_lightning as pl
import scprep
from matplotlib import pyplot as plt

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--protein_name', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--random_seed', type=int, default=1)
    # parse params
    args = parser.parse_args()

    npy_root = '../../output_numpy/'
    npy_path = '%s/%s/%s_%s_alpha=%s/' % (npy_root, args.protein_name,
                                          args.model, args.protein_name, args.alpha)
    plot_path = '../../output_figures/%s/' % (args.protein_name)

    # Reproducibility.
    pl.seed_everything(args.random_seed)

    data = np.load(npy_path + 'embeddings.npy')
    times = np.load(npy_path + 'times.npy') * 10  # now it is in micro sec.
    # times = np.load(npy_path + 'times.npy') * 2e-4  # now it is in micro sec??

    os.makedirs(plot_path, exist_ok=True)

    phate_op = phate.PHATE(random_state=args.random_seed,
                           n_components=3, verbose=True)
    Y_phate3d = phate_op.fit_transform(data)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    scprep.plot.rotate_scatter3d(Y_phate3d,
                                 figsize=(8, 6), cmap="Spectral",
                                 ticks=False, label_prefix="PHATE",
                                 c=times,
                                 ax=ax)

    fig.savefig(plot_path + 'PHATE_%s_%s_alpha=%s.png' %
                (args.model, args.protein_name, args.alpha))
