import pandas as pd
import numpy as np
import networkx as nx
import argparse
from datetime import datetime
from scipy.sparse.csgraph import connected_components

# python create_traces.py --eventlog multi_completed_cases.csv --activity_key ACTIVITY_NAME --time_key START_DATE_TIME --id_cols ID_CONTRACT_LINE ID_REQ_LINE ID_ORDER_LINE ID_RECEIPT_LINE ID_INVOICE_HEADER

parser = argparse.ArgumentParser()

parser.add_argument('--eventlog', default="sepsis_sample_raw.csv", type=str)
parser.add_argument('--id_cols', type=str, nargs='+', help='List of multiple case ID columns for object-centric datasets')
parser.add_argument('--activity_key', default="concept:name", type=str)
parser.add_argument('--time_key', default="time", type=str)
parser.add_argument('--time_format', default="%Y-%m-%d %H:%M:%S",
                        type=str) 

args = parser.parse_args()
start_time = datetime.now()

# first try to read it using; if it is not the right separator use the other (do it separately to avoid confusion)
df = pd.read_csv("../data/%s" % args.eventlog, sep=';')
if df.shape[1] == 1:
    df = pd.read_csv("../data/%s" % args.eventlog, sep=',', low_memory=False)

# If the column is a datetime cast it to seconds
if df[args.time_key].dtype == "object":
    df[args.time_key] = pd.to_datetime(df[args.time_key], format=args.time_format).astype(np.int64) / int(1e9)

# order process by timestamp
df = df.sort_values(args.time_key, ascending=True).reset_index(drop=True)
percentage = 0

# create unique adj matrix and from there find the connected components (each graph will be a different case)
# here we are not interested in the direction of the relationship, just that there is a connection in order to keep two elements together
G = nx.Graph()
for ids in zip(df.index, df["ID_CONTRACT_LINE"], df['ID_REQ_LINE'], df['ID_ORDER_LINE'], \
            df['ID_RECEIPT_LINE'], df['ID_INVOICE_HEADER']):
    # first add the node - each node as the unique id (event identifier) and the related attributes
    G.add_node(ids[0], attr_dict={"ID_CONTRACT_LINE": ids[1], 'ID_REQ_LINE': ids[2], 'ID_ORDER_LINE': ids[3], \
                                'ID_RECEIPT_LINE': ids[4], 'ID_INVOICE_HEADER': ids[5]})

    if ids[0] == df.index[-1]:
        # the last element has no more links
        break
    else:
        # extract all not nan ids for the current element and find each id in the successors
        ids_not_null = df.loc[ids[0]][args.id_cols].dropna().values
        id_names = df.loc[ids[0]][args.id_cols].dropna().index
        # we search only in the successors to reduce the search space
        df_successors = df.loc[ids[0]+1:].copy()
        # if there are successors (neighbours) for a given node
        neighbours = df_successors.loc[df_successors[id_names[0]] == ids_not_null[0]].index
        for neighbour in neighbours:
            G.add_edge(ids[0], neighbour, attr_dict=None)
        # if there are more ids to search, add also neighbours for the other id not already added
        if len(id_names) == 2:
            neighbours_second = df_successors.loc[df_successors[id_names[1]] == ids_not_null[1]].index
            neighbours_second = np.setdiff1d(neighbours_second, neighbours)
            for neighbour in neighbours_second:
                G.add_edge(ids[0], neighbour, attr_dict=None)
        
        if int(ids[0] % (len(df) / 10)) == 0:
            print("{} %".format(percentage))
            percentage += 10

# sorted is needed otherwise it messes with your nodes!
A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))

# find the components connected between each other (each component is a graph, i.e. the case) - for each element return its corresponding case id
trace_indicator = connected_components(A)
# trace_indicator = connected_components(np.asarray(A.todense()))
df["Case ID"] = trace_indicator[1]
# check for each set of case ids that there is never the same element for two different case ids
df = df.sort_values(["Case ID", args.time_key], ascending=True).drop_duplicates().reset_index(drop=True)
df.to_csv(f"../data/{args.eventlog.strip('.csv')}_connected_components.csv", index=False)
final_time = datetime.now() - start_time
print("Time to extract cases from original dataset: {}".format(final_time))