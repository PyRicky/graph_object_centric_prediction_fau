from spektral.data import Dataset, Graph
import pandas as pd
import numpy as np
import scipy.sparse as sp
from spektral.utils import normalized_adjacency
import networkx as nx
from datetime import datetime

class NormalizeAdj:
    r"""
        Normalizes the adjacency matrix as:
    $$
        \A \leftarrow \D^{-1/2}\A\D^{-1/2}
        $$

        **Arguments**

        - `symmetric`: If False, then it computes \(\D^{-1}\A\) instead.
    """

    def __init__(self, symmetric=True):
        self.symmetric = symmetric

    def __call__(self, graph):
        if graph.a is not None:
            graph.a = normalized_adjacency(graph.a, self.symmetric)

        return graph


class GOCPDataset(Dataset):
    """
    A dataset of process instances
    This one does use a, x, e for as features and y as labels
    """

    def __init__(self, args, object_case_ids, mean_events_per_case, df=None, transform=None, **kwargs):
        self.args = args
        self.object_case_ids = object_case_ids
        self.transform = transform
        self.mean_events_per_case = mean_events_per_case
        if df is not None:
            self.df = self.apply_edges_to_df(df)
            self.cases = self.df["case:concept:name"].unique()
            self.object_case_ids["case:concept:name"] = self.df["case:concept:name"]
            start_time = datetime.now()
            # for each case we calculate the adjacency matrix - this was the first version, always forward
            self.adj_matrixes = self.df.groupby("case:concept:name").count()["concept:name_1"].apply(self.calculate_adj_matrixes)
            # self.adj_matrixes = self.object_case_ids.groupby("case:concept:name").apply(self.calculate_adj_matrixes_connected_ids)
            time = datetime.now() - start_time
            print("Time to calculate Adj matrixes: {}".format(time))
            self.graphs = self.read()
        else:
            # this is needed when we manually put the graphs for the batch
            self.graphs = []
        # super().__init__(**kwargs)

    def apply_edges_to_df(self, df):
        start_time = datetime.now()
        original_activity_column = pd.DataFrame(df["concept:name"])
        original_activity_column["case:concept:name"] = df["case:concept:name"]

        if self.args.id_cols:
            original_activity_column[self.args.id_cols] = self.object_case_ids[self.args.id_cols]
            # for the object centric case the edge type is backward or recursive only if the activity happens 2+ times 
            # for the same set of multiple case ids (it's recursive iff 01(B) -> 01(B))
            columns_to_filter_edge_types = self.args.id_cols.copy()
            columns_to_filter_edge_types.extend(["concept:name"])
        else:
            # in the traditional single case id log only the activity is needed to understand the type of edge
            columns_to_filter_edge_types = ["concept:name"]

        original_activity_column["edge_type"] = [[1, 0, 0]] * original_activity_column.shape[0]  # default forward

        original_activity_column["backward"] = original_activity_column.groupby("case:concept:name").apply(
            lambda x: x[columns_to_filter_edge_types].duplicated()).reset_index()[0]
        original_activity_column.loc[original_activity_column["backward"] == True, "edge_type"] = \
            pd.Series([[0, 1, 0]] * original_activity_column.shape[0])  # set backward edge type

        # set recursive edge type - match each event with the previous event
        current_df = original_activity_column[columns_to_filter_edge_types]
        previous_df = original_activity_column[columns_to_filter_edge_types].shift(1)
        # here matching nans won't constitute a difference (NaN = NaN)
        compare_df = current_df.compare(previous_df, keep_shape=True)
        # if a row is always nan it means the previous one was equal - therefore it is a recursive relationship
        recursive_edges = compare_df.index[compare_df.isnull().all(1)]
        original_activity_column.loc[original_activity_column.index.isin(recursive_edges), "edge_type"] = pd.Series(
            [[0, 0, 1]] * original_activity_column.shape[0])

        df["edge_type"] = original_activity_column["edge_type"]
        del df["concept:name"]
        time = datetime.now() - start_time
        print("Time to calculate edges: {}".format(time))
        return df


    def calculate_adj_matrixes_connected_ids(self, case_df):
        G = nx.DiGraph()
        for ids in zip(case_df.index, case_df["ID_CONTRACT_LINE"], case_df['ID_REQ_LINE'], case_df['ID_ORDER_LINE'], \
                    case_df['ID_RECEIPT_LINE'], case_df['ID_INVOICE_HEADER']):
            # first add the node - each node as the unique id (event identifier) and the related attributes
            G.add_node(ids[0], attr_dict={"ID_CONTRACT_LINE": ids[1], 'ID_REQ_LINE': ids[2], 'ID_ORDER_LINE': ids[3], \
                                        'ID_RECEIPT_LINE': ids[4], 'ID_INVOICE_HEADER': ids[5]})

            # the first node is the pseudo start, so you just have one link forward to the first real activity
            if ids[0] == case_df.index[0]:
                G.add_edge(ids[0], ids[0]+1, attr_dict=None)
            elif ids[0] == case_df.index[-1]:
                # the last element has no more links
                break
            else:
                # extract all not nan ids for the current element and find each id in the successors
                ids_not_null = case_df.loc[ids[0]][self.args.id_cols].dropna().values
                id_names = case_df.loc[ids[0]][self.args.id_cols].dropna().index
                # we search only in the successors to reduce the search space
                df_successors = case_df.loc[ids[0]+1:].copy() 
                neighbours = df_successors.loc[df_successors[id_names[0]] == ids_not_null[0]].index
                for neighbour in neighbours:
                    G.add_edge(ids[0], neighbour, attr_dict=None)
                # if there are more ids to search, add also neighbours for the other id not already added
                if len(id_names) == 2:
                    neighbours_second = df_successors.loc[df_successors[id_names[1]] == ids_not_null[1]].index
                    neighbours_second = np.setdiff1d(neighbours_second, neighbours)
                    for neighbour in neighbours_second:
                        G.add_edge(ids[0], neighbour, attr_dict=None)

        A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))
        return A.todense()
        # print(A.todense())
        # import matplotlib.pyplot as plt
        # nx.draw(G, with_labels=True)
        # plt.show()


    def calculate_adj_matrixes(self, n):
        # used undirected graph - it creates a diagonal of ones over the main diagonal
        a = np.diagflat(np.ones(n - 1, int), 1)  # forward direction
        # for now keep only the last 20 events of the prefix
        # TODO: when doing the "advanced" adj matrix, you will need to consider the relations between just the last 20 events
        # in this case the adj matrix remains the same, since you always go forward
        # a = np.diagflat(np.ones(min(self.mean_events_per_case-1, n - 1), int), 1)
        return a

    def read(self):

        def make_graph(case, prefix_size):
            """
            istance is the full instance of the process, 
            prefix is the partial trace we are considering  
            """

            instance = self.df.loc[self.df["case:concept:name"] == case]
            prefix = instance.iloc[0:prefix_size, :].copy()

            # Edge matrix (e = edges * edge features)
            if self.args.architecture == 'ECC' or self.args.architecture == 'GatedGraph':
                # we discard the first element (no previous event) - now we keep the last because it is not an end activity anymore
                e = np.array(prefix["edge_type"].iloc[1:prefix.shape[0] + 1].tolist())
                # if prefix.shape[0] <= self.mean_events_per_case:
                #     e = np.array(prefix["edge_type"].iloc[1:prefix.shape[0] + 1].tolist())
                # else:
                #     # try keeping just the last n-1 edges
                #     e = np.array(prefix["edge_type"].iloc[-(self.mean_events_per_case-1):].tolist())
            del prefix["edge_type"]
            del prefix["case:concept:name"]

            # extract the y (target)
            if self.args.kpi == "next_activity":
                # here you have a vector with a 1 corresponding to the next activity to be predicted
                y = np.array(prefix['y'].iloc[-1])
            else:
                # here you have a number to be predicted 
                y = prefix['y'].iloc[-1]
            del prefix['y']

            # Node matrix (x = nodes * node features) - e.g. with prefix 2 the shape will be (2, n_features)
            x = np.array(prefix.values.tolist())
            # now we try keeping only the features of the last 20 events of the prefix in order to not have too many features
            # x = np.array(prefix.iloc[-self.mean_events_per_case:].values.tolist())

            # Adjacency matrix (a = nodes * nodes)
            # it takes the adj matrix related to the case and then it cuts it according to the prefix dimension (n*n)
            a = np.split(np.split(self.adj_matrixes[case], [len(prefix)])[0], [len(prefix)], axis=1)[0]
            a = sp.csr_matrix(a)

            if self.args.architecture == 'ECC' or self.args.architecture == 'GatedGraph':
                # graph = NormalizeAdj()(Graph(x=x, a=a, y=y))
                graph = NormalizeAdj()(Graph(x=x, a=a, y=y, e=e))
                # if self.transform is not None:
                #     graph = NormalizeAdj()(graph)
                return graph
                # return Graph(x=x, a=a, y=y, e=e)
            else:
                graph = NormalizeAdj()(Graph(x=x, a=a, y=y))
                return graph
                # return Graph(x=x, a=a, y=y)

        # Note: We must return a list of Graph objects. It only takes into account traces > len(2).
        # +1 because we include also the last event now (we have already deleted the end activities from each trace)
        # cambia list comprehension in for and then yield make graph
        # return [make_graph(case, prefix_size) for case in self.cases for
        #         prefix_size in range(2, len(self.df.loc[self.df["case:concept:name"] == case]) + 1)]

        for case in self.cases:
            for prefix_size in range(2, len(self.df.loc[self.df["case:concept:name"] == case]) + 1):
                yield make_graph(case, prefix_size)


def create_graph_chunks_and_save_to_file(train_indexes, val_indexes, test_indexes, graphs, batch_size):
    ''' 
    Takes as input the indexes of the graphs for train/test/valid, for each index (prefix) solves the generator containing the graph 
    and saves the real graphs in the filesystem.
    '''

    import time, pickle
    from datetime import timedelta
    start_time = time.time()
    print("Creating graphs for all prefixes ...")
        
    i = 0
    while(True):
        graph = next(graphs, None)
        if (int(i % (batch_size*100)) == 0):
            # write progress every 100 batches
            print(f"Written {int(i/batch_size)} batches")
        if graph == None:
            break
        else:
            if i in train_indexes:
                with open(f'../graph_chunks/train_batch_{i}.pickle', 'wb') as handle:
                    pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
            elif i in val_indexes:
                with open(f'../graph_chunks/valid_batch_{i}.pickle', 'wb') as handle:
                    pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
            elif i in test_indexes:
                with open(f'../graph_chunks/test_batch_{i}.pickle', 'wb') as handle:
                    pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
            i += 1

    print(f"Time to create graphs: {str(timedelta(seconds=round(time.time() - start_time)))}")