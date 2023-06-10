import pickle, gc, shutil, os
from os.path import join
import numpy as np
import pandas as pd
import tensorflow as tf
from spektral.data import DisjointLoader
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.transforms.gcn_filter import GCNFilter
from tensorflow.python.keras.initializers.initializers_v2 import GlorotUniform
from spektral.layers import GatedGraphConv, GlobalAttentionPool
from tensorflow.python.ops.array_ops import zeros
from gocp.src.config import load
from gocp.src.preprocessing import common
from gocp.src.preprocessing.preprocessor import encode_df, encode_case_ids_as_node_features, \
    clean_columns_and_reduce_categorical_domain_dynamic, get_activity_id_mapping
from gocp.src.preprocessing.encoding import Encoding
from datetime import datetime
from random import shuffle
from graph_dataset import GOCPDataset, create_graph_chunks_and_save_to_file
from train_evaluate_model import Model

# gpus = tf.config.list_physical_devices('GPU')
# print(len(gpus), "Physical GPUs,", len(tf.config.experimental.list_logical_devices('GPU')), "Logical GPU")

# PARAMETERS
args = load()
eventlog = args.eventlog
time_format = args.time_format
experiment_folder = "experiment_files"

CONTEXT_ATTRIBUTE_NAME = args.first_context_attr_name

learning_rate = args.learning_rate
epochs = args.epochs  # Number of training epochs
es_patience = args.es_patience  # Patience for early stopping
batch_size = args.batch_size
if args.seed:  # Make weight initialization reproducible
    np.random.seed(seed=args.seed_val)
    tf.random.set_seed(seed=args.seed_val)

# Folders for IO
if not os.path.exists('../graph_chunks'):
    os.makedirs(('../graph_chunks'), exist_ok=not True)
if not os.path.exists("../experiments"):
    os.makedirs("../experiments", exist_ok=not True)
if not os.path.exists(f"../{experiment_folder}"):
    os.makedirs(f"../{experiment_folder}", exist_ok=not True)
    os.makedirs(f"../{experiment_folder}/model", exist_ok=not True)
    os.makedirs(f"../{experiment_folder}/result", exist_ok=not True)

# clean results from previous experiments
[os.remove(f"../{experiment_folder}/model/{file}") for file in os.listdir(f"../{experiment_folder}/model/")]
[os.remove(f"../{experiment_folder}/result/{file}") for file in os.listdir(f"../{experiment_folder}/result/")]
[os.remove(f"../graph_chunks/{chunk}") for chunk in os.listdir(f"../graph_chunks/")]

# MAIN SCRIPT
# first try to read it using; if it is not the right separator use the other (do it separately to avoid confusion)
df = pd.read_csv("../data/%s" % eventlog, sep=';')
if df.shape[1] == 1:
    df = pd.read_csv("../data/%s" % eventlog, sep=',', low_memory=False)

# If the column is a datetime cast it to seconds
if df[args.time_key].dtype == "object":
    df[args.time_key] = pd.to_datetime(df[args.time_key], format='%d.%m.%Y-%H:%M:%S').astype(np.int64) / int(1e9)

source_entity = common.calculate_source_entity(args)
if args.source_activity and args.target_activity:
    df = common.select_path(args, df)

# def apply_history_to_df(df):
#     """add how many times a particular activity has been performed"""
#     ohe_activities = pd.get_dummies(df["ACTIVITY_NAME"], prefix="# Activity", prefix_sep='=')
#     df_ohe = ohe_activities.join(df["Case ID"])
#     df_ohe = df_ohe.groupby("Case ID")[ohe_activities.columns].cumsum()
#     df = df.join(df_ohe)
#     print("Added aggregated history")
#     return df

# df = apply_history_to_df(df)

if source_entity is not None and eventlog != "df_connected_components_concurrent.csv":
    df = clean_columns_and_reduce_categorical_domain_dynamic(df, args, source_entity)
df = common.add_time_features_to_dataframe(df, args)

if args.add_pseudo_start:
    df = common.add_pseudo_activity(args=args, df=df)

encoder = Encoding()
renamed_df = encoder.rename_csv_columns(args, df)
# needed for the next activity prediction - get association activity name vs ohe value
ohe_mapping, activity_id_name_mapping = get_activity_id_mapping(renamed_df)

original_activity_column = renamed_df["concept:name"]
renamed_df = encode_df(renamed_df, args)

if args.id_cols:
    # it is needed later for the edge calculation in the object-centric case
    object_case_ids = df[args.id_cols]
    renamed_df, df_result_case_ids, object_names = encode_case_ids_as_node_features(renamed_df, original_activity_column, args)
else:
    object_case_ids = None

renamed_df, df, kpi_type, kpi_in_column, object_case_ids = common.calculate_target_kpi(args, df, renamed_df, object_case_ids)
# del df
# gc.collect()
common.print_kpi_statistics(args, renamed_df, kpi_type)

# we calculate the mean_events per case just to keep the "essential" history
df["Prefix"] = df.groupby("Case ID").cumcount()+1
cases_length = df.groupby("Case ID")["Prefix"].max()
mean_events_per_case = int(cases_length.mean())
del df["Prefix"]

df, renamed_df = common.visualize_case_distribution_and_remove_skewed_cases(df, renamed_df, kpi_type, remove_skewed_cases=False)

# insert this column again (needed later to easily calculate the edges)
renamed_df["concept:name"] = original_activity_column
del renamed_df["time:timestamp"]

# we just keep the essential features for the GNN and the GCN, since the results have shown that accuracy would worsen a lot
#  features), later we should extend this
cols = [col for col in renamed_df if
        col.startswith('USER') and not col.startswith('USER_TYPE') and not col.startswith(
            'USER_CATEGORY') or col.startswith('concept:name') or col.startswith('#')]
cols.extend(
    ["case:concept:name", "time_from_previous_event", "time_from_start", "time_from_midnight", "weekday"])
cols.extend(object_names)
cols.append("y")

renamed_df = renamed_df[cols]

dataset = GOCPDataset(args=args, object_case_ids=object_case_ids, mean_events_per_case=mean_events_per_case, df=renamed_df, transforms=NormalizeAdj(True)) 

n_node_features = renamed_df.shape[1] - 3 # Dimension of node features
n_edge_features = len(renamed_df["edge_type"][0]) # Dimension of edge features
if type(renamed_df["y"][0]) == list:
    n_out = len(renamed_df["y"][0]) # Dimension of the target
else:
    n_out = 1 

# TODO: current version is shuffling the cases (to replicate the older paper) - do we want to take them in order?
train_indexes, val_indexes, test_indexes = common.train_test_split(df, experiment_folder)
# we use the df (removing the pseudo_start as the graphs) to understand later the results per prefix
first_indexes = df.groupby('Case ID',as_index=False).nth(0).index
df = df.loc[~df.index.isin(first_indexes)].reset_index(drop=True)
cols = [args.case_id_key, args.activity_key]
cols.extend(args.id_cols)
df = df[cols]
create_graph_chunks_and_save_to_file(train_indexes, val_indexes, test_indexes, dataset.graphs, batch_size)

del dataset
gc.collect()

model = Model(args, n_node_features, n_edge_features, n_out, kpi_type, kpi_in_column, batch_size, ohe_mapping, activity_id_name_mapping, learning_rate)

# FIT COMPLETE MODEL
start_training_time = datetime.now()
current_batch = epoch = model_loss = model_acc = 0
best_val_loss = np.inf
best_weights = None
patience = es_patience
history = {}

# instantiate a new GOCPDataset class, which will store each time 32 graphs read from memory (batch)
if args.architecture == "GCN":
    dataset_batch = GOCPDataset(args=args, object_case_ids=object_case_ids, mean_events_per_case=mean_events_per_case, transforms=GCNFilter(True))
else:
    dataset_batch = GOCPDataset(args=args, object_case_ids=object_case_ids, mean_events_per_case=mean_events_per_case, transforms=NormalizeAdj(True))

if not os.path.exists(f"../{experiment_folder}/model/model.h5"):
    for epoch in range(epochs):
        # At the start of each epoch, shuffle all the graphs
        shuffle(train_indexes)
        batch_count = 0
        train_batches = 0
        # access all the graphs, and then fit one batch at a time
        for i in range(0, len(train_indexes)):
            with open(f'../graph_chunks/train_batch_{train_indexes[i]}.pickle', 'rb') as handle:
                dataset_batch.graphs.append(pickle.load(handle))
            batch_count += 1
            # if we have enough elements for a batch or it is the last step and the number of samples is not divisible by the batch size
            if (batch_count == batch_size) or (((i+1) == len(train_indexes)) and (batch_count != 0)):
                loader_tr = DisjointLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                for batch in loader_tr: # this for is just for expanding the variable, since the batch is one already
                    outs = model.train_step(*batch)
                    model_loss += outs
                    break
                dataset_batch.graphs = []
                batch_count = 0
                train_batches += 1


        model_loss /= train_batches
        val_loss = model.evaluate_model(dataset_batch, val_indexes, batch_size, "valid", df, experiment_folder) # Compute validation loss
        print("Ep. {} - Loss: {:.2f} - Val loss: {:.2f}".format(epoch, float(model_loss), float(val_loss)))
        history[epoch] = {'loss': float(model_loss), 'val_loss': float(val_loss)}

        if val_loss < best_val_loss:  # Check if loss improved for early stopping
            best_val_loss = val_loss
            patience = es_patience
            print("New best val_loss {:.3f}".format(float(val_loss)))
            best_weights = model.model.get_weights()
            # save the best model
            model.model.save(f"../{experiment_folder}/model/model.h5")

        else:
            patience -= 1
            if patience == 0:
                print("Early stopping (best val_loss: {})".format(best_val_loss))
                break
        model_loss = 0

    common.plot_training_loss(args, kpi_in_column, history, kpi_type, experiment_folder)
    training_time = datetime.now() - start_training_time
    model.model.set_weights(best_weights)  # Load best model
    print("Training time: {}".format(training_time))

else:
    from tensorflow.keras.initializers import zeros
    # this is needed otherwise it doesn't recognize certain layers
    custom_object = {"GatedGraphConv": GatedGraphConv, "GlobalAttentionPool": GlobalAttentionPool, \
    "GlorotUniform": GlorotUniform, "Zeros": zeros}
    model.model = tf.keras.models.load_model(f'../{experiment_folder}/model/model.h5', custom_objects=custom_object)
    print('Reloaded model')

# now obtain the scores of this model over the validation set in order to compute statistics per prefix (to be compared with GNN and decide which model to take on the test set)
model.evaluate_model(dataset_batch, val_indexes, batch_size, "valid", df, experiment_folder, compare=True)

# EVALUATE MODEL
print("Testing model")
test_loss = model.evaluate_model(dataset_batch, test_indexes, batch_size, "test", df.loc[test_indexes], experiment_folder)
print("Done. Test loss: {:.4f}.".format(test_loss[0]))

# save results and remove previous saved model
fromDirectory = f"../{experiment_folder}"
toDirectory = f'../experiments/{args.experiment_name}'
shutil.copytree(fromDirectory, toDirectory)