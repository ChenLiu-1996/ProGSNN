import os
import pickle
from argparse import ArgumentParser
from functools import partial
from glob import glob

from graphein.ml.conversion import GraphFormatConvertor
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import add_k_nn_edges
from graphein.protein.graphs import construct_graph
from tqdm import tqdm

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--protein_name', type=str, required=True)
    args = parser.parse_args()

    print("getting graphs")
    data_root = '/gpfs/gibbs/pi/krishnaswamy_smita/de_shaw/%s/' % args.protein_name
    output_dir = '../../input_graphs/%s/' % args.protein_name
    os.makedirs(output_dir, exist_ok=True)

    time_ranges = [path.split('/')[-1] for path in glob(data_root + '*')]

    range_to_graphname = {}
    for time_range in time_ranges:
        t_start = time_range.split(' to ')[0]
        t_end = time_range.split(' to ')[1].split(' us')[0]
        range_to_graphname[time_range] = '%sto%sgraphs' % (t_start, t_end)

    for time_range in time_ranges:
        # Override config with constructors
        constructors = {
            'edge_construction_functions':
            [partial(add_k_nn_edges, k=3, long_interaction_threshold=0)],
            'pdb_dir':
            data_root + time_range,
            # "edge_construction_functions": [add_hydrogen_bond_interactions, add_peptide_bonds],
            # "node_metadata_functions": [add_dssp_feature]
        }

        config = ProteinGraphConfig(**constructors)

        # Make graphs
        graph_list = []
        y_list = []
        pdb_paths = glob(data_root + time_range + '/*')

        for idx, pdb_path in enumerate(tqdm(pdb_paths)):
            graph_list.append(construct_graph(
                pdb_path=pdb_path, config=config))

        format_convertor = GraphFormatConvertor('nx',
                                                'pyg',
                                                verbose='gnn',
                                                columns=None)

        pyg_list = [format_convertor(graph) for graph in tqdm(graph_list)]

        with open('%s/%s.pkl' % (output_dir, range_to_graphname[time_range]),
                  'wb') as file:
            pickle.dump(pyg_list, file)

    print("combining data")
    total_graphs = []
    existing_outputs = os.listdir(output_dir)
    for i, output_fname in enumerate(tqdm(existing_outputs)):
        split1, split2 = output_fname.split('.')

        if split2 == 'pkl' and split1 != 'total_graphs':

            with open(output_dir + output_fname, "rb") as file:
                graphs = pickle.load(file)
                for graph in graphs:
                    total_graphs.append(graph)

    with open(output_dir + 'total_graphs.pkl', 'wb') as out:
        pickle.dump(total_graphs, out)
