from __future__ import division, print_function

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class DEShaw(Dataset):
    """ZINCTranch dataset."""

    def __init__(self, file_name, transform=None):
        with open(file_name, 'rb') as file:
            self.graphs = pickle.load(file)

        # Record all the unique amino acids.
        self.amino_acid_dict = {}
        for graph in self.graphs:
            node_ids = graph.node_id
            for i in range(graph.num_nodes):
                acid = node_ids[i]
                _, acid_abbv, _ = acid.split(':')
                if acid_abbv not in self.amino_acid_dict:
                    self.amino_acid_dict[acid_abbv] = len(self.amino_acid_dict)

        self.num_node_features = len(self.amino_acid_dict.keys())
        self.transform = transform

        self.files_by_folder = {}
        for graph in self.graphs:
            folder_id, file_id = graph.name[0].split('_')[1].split('-')
            file_id = int(file_id)
            if folder_id not in self.files_by_folder:
                self.files_by_folder[folder_id] = [file_id]
            else:
                self.files_by_folder[folder_id].append(file_id)
        for folder_id in self.files_by_folder.keys():
            self.files_by_folder[folder_id] = sorted(
                self.files_by_folder[folder_id])

    def __len__(self):

        return len(self.graphs)

    def __getitem__(self, idx):

        graph = self.graphs[idx]
        nodes = graph.node_id

        features = []
        for i in range(graph.num_nodes):
            acid_presence = np.zeros(len(self.amino_acid_dict))
            acid = nodes[i]
            _, acid_abbv, _ = acid.split(':')
            acid_idx = self.amino_acid_dict[acid_abbv]
            acid_presence[acid_idx] = 1
            features.append(acid_presence)

        graph.x = torch.tensor(np.array(features)).float()
        graph.edge_attr = None

        # Get the time information.
        folder_id, file_id = graph.name[0].split('_')[1].split('-')
        # folder_us are in 2 micro sec intervals.
        # file_id are sequential.
        file_loc = self.files_by_folder[folder_id].index(int(file_id))
        time_us = int(folder_id)
        time_sub_us = file_loc / len(self.files_by_folder[folder_id])
        time = 2 * (time_us + time_sub_us)
        # `time` is in 0-10 micro sec. Normalize time to [0, 1].
        time = time / 10

        graph.time = time

        if self.transform:
            return self.transform(graph)
        else:
            return graph
