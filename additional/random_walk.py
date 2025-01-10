# walk strategy: deep walk, node2vec, metapath2vec
# input: graph (networkx)
import pickle
import random
import numpy as np
import os
from collections import defaultdict

import scipy.sparse as sp
from joblib import Parallel, delayed, load, dump
from tqdm import tqdm
import networkx


class RandomWalk:
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    def __init__(self, graph, walk_length=3, num_walks=500, p=1, q=1,
                 weight_key='weight',
                 workers=1, sampling_strategy=None, quiet=False, temp_folder=None,
                 optp=False, source_node=None, not_pre=False,
                 ):
        """
        Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.

        :param graph: Input graph
        :type graph: Networkx Graph
        :param walk_length: Number of nodes in each walk (default: 80)
        :type walk_length: int
        :param num_walks: Number of walks per node (default: 10)
        :type num_walks: int
        :param p: Return hyper parameter (default: 1)
        :type p: float
        :param q: Inout parameter (default: 1)
        :type q: float
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :type weight_key: str
        :param workers: Number of workers for parallel execution (default: 1)
        :type workers: int
        # :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        :param sampling_strategy: Only the nodes included in the sampling_strategy are going to be generated.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        :param temp_folder: Path to folder with enough space to hold the memory map of self.d_graph (for big graphs); to be passed joblib.Parallel.temp_folder
        :type temp_folder: str
        """

        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.workers = workers
        self.quiet = quiet
        self.d_graph = defaultdict(dict)

        if sampling_strategy is None:
            self.sampling_strategy = set()
        else:
            self.sampling_strategy = sampling_strategy

        if source_node is not None:
            self.source_node = source_node
        else:
            self.source_node = sampling_strategy

        self.temp_folder, self.require = None, None
        if temp_folder:
            if not os.path.isdir(temp_folder):
                raise NotADirectoryError(
                    "temp_folder does not exist or is not a directory. ({})".format(
                        temp_folder))

            self.temp_folder = temp_folder
            self.require = "sharedmem"

        if not not_pre:
            self._precompute_probabilities_opt() if optp else self._precompute_probabilities()
            self.walks = self._generate_walks()
        else:
            self.walks = self._generate_walks_without_precompute()

    def _generate_walks_without_precompute(self):
        """
        Generates the random walks
        """
        # prams
        quiet = self.quiet
        num_walks = self.num_walks
        global_walk_length = self.walk_length
        d_graph = self.d_graph
        sampling_strategy = self.source_node
        neighbors_key = self.NEIGHBORS_KEY
        probabilities_key = self.PROBABILITIES_KEY
        first_travel_key = self.FIRST_TRAVEL_KEY
        num_walks_key = self.NUM_WALKS_KEY
        walk_length_key = self.WALK_LENGTH_KEY
        cpu_num = 0

        walks = list()

        if not quiet:
            pbar = tqdm(total=num_walks,
                        desc='Generating walks (CPU: {})'.format(cpu_num))

        for n_walk in range(num_walks):

            # Update progress bar
            if not quiet:
                pbar.update(1)

            # Shuffle the nodes
            # shuffled_nodes = list(d_graph.keys())
            shuffled_nodes = list(sampling_strategy)
            random.shuffle(shuffled_nodes)

            # Start a random walk from every node
            for source in shuffled_nodes:

                # Skip nodes with specific num_walks
                # if source in sampling_strategy and \
                #     num_walks_key in sampling_strategy[source] and \
                #     sampling_strategy[source][num_walks_key] <= n_walk:
                #     continue

                # Start walk
                walk = [source]

                # Calculate walk length
                # if source in sampling_strategy:
                #     walk_length = sampling_strategy[source].get(
                #         walk_length_key, global_walk_length)
                # else:
                #     walk_length = global_walk_length
                walk_length = global_walk_length

                # Perform walk
                while len(walk) < walk_length:

                    self.generate_walk_options(d_graph, walk[-1])

                    walk_options = d_graph[walk[-1]].get(neighbors_key, None)

                    # Skip dead end nodes
                    if not walk_options:
                        break

                    if len(walk) == 1:  # For the first step
                        self.generate_probabilities_first_step(d_graph, walk[-1],
                                                               first_travel_key,
                                                               walk_options)
                        probabilities = d_graph[walk[-1]][first_travel_key]
                        # if np.isnan(probabilities).any():
                        #     print('nan in probabilities')
                        walk_to = np.random.choice(
                            walk_options, size=1, p=probabilities)[0]
                    else:
                        self.generate_probabilities(d_graph, walk[-1],
                                                    probabilities_key, walk[-2],
                                                    walk_options)
                        probabilities = d_graph[walk[-1]
                        ][probabilities_key][walk[-2]]
                        # if the probabilities contain nan, we should skip this node
                        # if np.isnan(probabilities).any():
                        #     print('nan in probabilities')
                        walk_to = np.random.choice(
                            walk_options, size=1, p=probabilities)[0]

                    walk.append(walk_to)

                walk = list(map(int, walk))  # Convert all to strings

                walks.append(walk)

        if not quiet:
            pbar.close()

        return walks

    def generate_walk_options(self, d_graph, node):
        if self.NEIGHBORS_KEY not in d_graph[node]:
            d_graph[node][self.NEIGHBORS_KEY] = list(self.graph.neighbors(node))

    def generate_probabilities_first_step(self, d_graph, node, first_travel_key,
                                          walk_options
                                          ):
        if first_travel_key not in d_graph[node]:
            # unnormalized_weights = [1] * len(walk_options)
            # consider weight now
            unnormalized_weights = [self.graph[node][neighbor].get(self.weight_key, 1)
                                    for neighbor in walk_options]
            unnormalized_weights = np.array(unnormalized_weights)
            d_graph[node][first_travel_key] = unnormalized_weights / unnormalized_weights.sum()

    def generate_probabilities(self, d_graph, node, probabilities_key, last_node,
                               walk_options
                               ):
        if probabilities_key not in d_graph[node]:
            d_graph[node][probabilities_key] = {}
        if last_node not in d_graph[node][probabilities_key]:
            # unnormalized_weights = [1] * len(walk_options)
            # unnormalized_weights = np.array(unnormalized_weights)
            # d_graph[node][probabilities_key][last_node] = unnormalized_weights / len(
            #     walk_options)
            if self.p == 1 and self.q == 1:
                self.generate_probabilities_first_step(d_graph, node, self.FIRST_TRAVEL_KEY,
                                                         walk_options)
                unnormalized_weights = d_graph[node][self.FIRST_TRAVEL_KEY]
                d_graph[node][probabilities_key][last_node] = unnormalized_weights
            else:
                unnormalized_weights = []
                for neighbor in walk_options:
                    if neighbor == last_node:
                        ss_weight = self.graph[node][neighbor].get(
                            self.weight_key, 1) * 1 / self.p
                    elif neighbor in self.graph[last_node]:
                        ss_weight = self.graph[node][neighbor].get(
                            self.weight_key, 1)
                    else:
                        ss_weight = self.graph[node][neighbor].get(
                            self.weight_key, 1) * 1 / self.q
                    unnormalized_weights.append(ss_weight)
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[node][probabilities_key][last_node] = unnormalized_weights / unnormalized_weights.sum()


    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """

        d_graph = self.d_graph
        first_travel_done = set()

        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')

        for source in nodes_generator:

            # Init probabilities dict for first travel
            if self.PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][self.PROBABILITIES_KEY] = dict()

            for current_node in self.graph.neighbors(source):

                # Init probabilities dict
                if self.PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][self.PROBABILITIES_KEY] = dict()

                unnormalized_weights = list()
                first_travel_weights = list()
                d_neighbors = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):

                    p = self.sampling_strategy[current_node].get(self.P_KEY,
                                                                 self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY,
                                                                 self.q) if current_node in self.sampling_strategy else self.q

                    if destination == source:  # Backwards probability
                        ss_weight = self.graph[current_node][destination].get(
                            self.weight_key, 1) * 1 / p
                    # If the neighbor is connected to the source
                    elif destination in self.graph[source]:
                        ss_weight = self.graph[current_node][destination].get(
                            self.weight_key, 1)
                    else:
                        ss_weight = self.graph[current_node][destination].get(
                            self.weight_key, 1) * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    if current_node not in first_travel_done:
                        first_travel_weights.append(
                            self.graph[current_node][destination].get(self.weight_key,
                                                                      1))
                    d_neighbors.append(destination)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node][self.PROBABILITIES_KEY][
                    source] = unnormalized_weights / unnormalized_weights.sum()

                if current_node not in first_travel_done:
                    unnormalized_weights = np.array(first_travel_weights)
                    d_graph[current_node][
                        self.FIRST_TRAVEL_KEY] = unnormalized_weights / \
                                                 unnormalized_weights.sum()
                    first_travel_done.add(current_node)

                # Save neighbors
                d_graph[current_node][self.NEIGHBORS_KEY] = d_neighbors

    def _precompute_probabilities_opt(self):
        """
        Precomputes transition probabilities for each node assuming edge weight is 1, for efficiency.
        assumes that p and q are 1
        """
        # nodes_generator = self.graph.nodes() if self.quiet else tqdm(self.graph.nodes(),
        #                                                              desc='Computing transition probabilities')
        d_graph = self.d_graph
        self.first_travel_done = set()
        if len(self.sampling_strategy) == 0:
            self.sampling_strategy = {node for node in self.graph.nodes()}
        nodes_generator = sorted(list(self.sampling_strategy))
        if not self.quiet:
            nodes_generator = tqdm(nodes_generator,
                                   desc='Computing transition probabilities')

        for source in nodes_generator:
            if self.PROBABILITIES_KEY not in self.d_graph[source]:
                self.d_graph[source][self.PROBABILITIES_KEY] = {}

            for current_node in self.graph.neighbors(source):
                if self.PROBABILITIES_KEY not in self.d_graph[current_node]:
                    self.d_graph[current_node][self.PROBABILITIES_KEY] = {}

                # Precompute factors for weight adjustments
                # p = self.sampling_strategy.get(current_node, {}).get(self.P_KEY, self.p)
                # q = self.sampling_strategy.get(current_node, {}).get(self.Q_KEY, self.q)

                # unnormalized_weights = []
                d_neighbors = list(self.graph.neighbors(current_node))

                # for destination in d_neighbors:
                #     # if destination == source:
                #     #     ss_weight = 1 / p
                #     # elif destination in self.graph[source]:
                #     #     ss_weight = 1
                #     # else:
                #     #     ss_weight = 1 / q
                #     ss_weight = 1
                #     unnormalized_weights.append(ss_weight)
                unnormalized_weights = [1] * len(d_neighbors)

                # Normalize and save probabilities
                unnormalized_weights = np.array(unnormalized_weights)
                self.d_graph[current_node][self.PROBABILITIES_KEY][
                    source] = unnormalized_weights / len(d_neighbors)

                # First travel is identical for all edges given equal weights
                if current_node not in self.first_travel_done:
                    self.d_graph[current_node][self.FIRST_TRAVEL_KEY] = np.ones(
                        len(d_neighbors)) / len(d_neighbors)
                    self.first_travel_done.add(current_node)

                self.d_graph[current_node][self.NEIGHBORS_KEY] = d_neighbors

    def _generate_walks(self):
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        def flatten(l): return [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers, temp_folder=self.temp_folder,
                                require=self.require)(
            delayed(parallel_generate_walks)(self.d_graph,
                                             self.walk_length,
                                             len(num_walks),
                                             idx,
                                             # self.sampling_strategy,
                                             self.source_node,
                                             self.NUM_WALKS_KEY,
                                             self.WALK_LENGTH_KEY,
                                             self.NEIGHBORS_KEY,
                                             self.PROBABILITIES_KEY,
                                             self.FIRST_TRAVEL_KEY,
                                             self.quiet) for
            idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)

        return walks

    def convert_walks_to_graphs(self):
        """
        Converts a list of walks to a list of networkx graph objects,
        which represents different orders of neighborhoods of nodes.
        :return: a list of networkx Graph
        """
        init_graph = networkx.from_scipy_sparse_matrix(
            sp.csr_matrix((len(self.graph), len(self.graph))),
            create_using=networkx.DiGraph())
        graphs = [networkx.DiGraph(init_graph) for _ in range(self.walk_length - 1)]
        for walk in self.walks:
            if len(walk) < self.walk_length:
                continue
            node = walk[0]
            for i in range(self.walk_length - 1):
                # if (node, walk[i + 1]) in graphs[i].edges()
                # we should add weight to the edge
                if graphs[i].has_edge(node, walk[i + 1]):
                    if self.WEIGHT_KEY not in graphs[i][node][walk[i + 1]]:
                        graphs[i][node][walk[i + 1]][self.WEIGHT_KEY] = 1
                    graphs[i][node][walk[i + 1]][self.WEIGHT_KEY] += 1
                else:
                    weight_attr = {self.WEIGHT_KEY: 1}
                    graphs[i].add_edge(node, walk[i + 1], **weight_attr)
        graphs = [networkx.to_scipy_sparse_matrix(graph, weight=self.WEIGHT_KEY,
                                                  nodelist=sorted(graph.nodes())
                                                  ) for
                  graph in graphs]
        # normalize the graphs
        # graphs = [self.normalize_sym(graph) for graph in graphs]
        return graphs

    def normalize_sym(self, adj):
        """Row-normalize sparse matrix"""
        rowsum = np.array(adj.sum(1)).astype(float)
        d_inv_sqrt = np.power(rowsum, -1).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj)


def parallel_generate_walks(d_graph, global_walk_length, num_walks, cpu_num,
                            sampling_strategy=None,
                            num_walks_key=None, walk_length_key=None,
                            neighbors_key=None, probabilities_key=None,
                            first_travel_key=None, quiet=False
                            ):
    """
    Generates the random walks which will be used as the skip-gram input.
    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks,
                    desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        if not quiet:
            pbar.update(1)

        # Shuffle the nodes
        # shuffled_nodes = list(d_graph.keys())
        shuffled_nodes = list(sampling_strategy)
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Skip nodes with specific num_walks
            # if source in sampling_strategy and \
            #     num_walks_key in sampling_strategy[source] and \
            #     sampling_strategy[source][num_walks_key] <= n_walk:
            #     continue

            # Start walk
            walk = [source]

            # Calculate walk length
            # if source in sampling_strategy:
            #     walk_length = sampling_strategy[source].get(
            #         walk_length_key, global_walk_length)
            # else:
            #     walk_length = global_walk_length
            walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:

                walk_options = d_graph[walk[-1]].get(neighbors_key, None)

                # Skip dead end nodes
                if not walk_options:
                    break

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key]
                    # if np.isnan(probabilities).any():
                    #     print('nan in probabilities')
                    walk_to = np.random.choice(
                        walk_options, size=1, p=probabilities)[0]
                else:
                    probabilities = d_graph[walk[-1]
                    ][probabilities_key][walk[-2]]
                    # if the probabilities contain nan, we should skip this node
                    # if np.isnan(probabilities).any():
                    #     print('nan in probabilities')
                    walk_to = np.random.choice(
                        walk_options, size=1, p=probabilities)[0]

                walk.append(walk_to)

            walk = list(map(int, walk))  # Convert all to strings

            walks.append(walk)

    if not quiet:
        pbar.close()

    return walks


def load_graph(path):
    """
    Load a graph from file
    :param path: The path to the edge list
    :return: networkx graph
    """
    with open(os.path.join(path, 'edges.pkl'), 'rb') as f:
        edges = pickle.load(f)
    adj_orig = sum([sub_adj.astype(np.float32) for sub_adj in edges])
    adj_orig += sp.eye(adj_orig.shape[0], dtype=np.float32)
    adj_orig[adj_orig > 0] = 1
    # adj_orig = sp.csr_matrix(adj_orig)
    G = networkx.from_scipy_sparse_matrix(adj_orig)
    return G


if __name__ == '__main__':
    data_path = "../data/IMDB"
    graph = load_graph(data_path)
    print('graph loaded')
    rw = RandomWalk(graph, walk_length=4, num_walks=1000)
    print('walks generated')
    graphs = rw.convert_walks_to_graphs()
    print('graphs generated')
    pickle.dump(graphs, open(os.path.join(data_path, 'walks.pkl'), 'wb'))
    print('walks saved')
