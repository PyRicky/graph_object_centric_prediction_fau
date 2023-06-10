import shutil, os, json
from os.path import join, exists
import numpy as np
import pandas as pd
import tensorflow as tf
from gocp.src.config import load
from gocp.src.preprocessing import common
from gocp.src.preprocessing.preprocessor import encode_df, encode_case_ids_as_node_features, \
    clean_columns_and_reduce_categorical_domain_dynamic, get_activity_id_mapping
from gocp.src.preprocessing.encoding import Encoding
from datetime import datetime
from datetime import timedelta

from keras.models import model_from_json
from lstm_dataset import prepare_data_for_lstm, train_model_generator
import time

from sklearn.metrics import classification_report, mean_absolute_error, f1_score

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

if source_entity is not None and "concurrent" not in eventlog:
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
common.print_kpi_statistics(args, renamed_df, kpi_type)

df, renamed_df = common.visualize_case_distribution_and_remove_skewed_cases(df, renamed_df, kpi_type, remove_skewed_cases=False)

del renamed_df["time:timestamp"]

# we commented this because we keep all the features
# cols = [col for col in renamed_df if
#         col.startswith('USER') and not col.startswith('USER_TYPE') and not col.startswith(
#             'USER_CATEGORY') or col.startswith('concept:name') or col.startswith('#')]
# cols.extend(
#     ["case:concept:name", "time_from_previous_event", "time_from_start", "time_from_midnight", "weekday",
#     "CONTRACT", "REQ", "ORDER", "RECEIPT", "INVOICE", "y"])
# renamed_df = renamed_df[cols]

mode = "train"
df["Prefix"] = df.groupby("Case ID").cumcount()+1
cases_length = df.groupby("Case ID")["Prefix"].max()
mean_events_per_case = round(cases_length.mean())
del df["Prefix"]

train_indexes, val_indexes, test_indexes = common.train_test_split(df, experiment_folder)

# drop from the dataframe the first prefix (we don't need the pseudo start and the indexes are already calculated without the pseudo start)
first_prefixes = df.groupby('Case ID',as_index=False).nth(0)
df = df.loc[~df.index.isin(first_prefixes.index)].reset_index(drop=True)
renamed_df = renamed_df.loc[~renamed_df.index.isin(first_prefixes.index)].reset_index(drop=True)

# since the train sequences will be a list "in order", we map the indexes in the dataset by resetting indexes, in order to avoid out of range error
dfTrain = renamed_df.loc[renamed_df.index.isin(train_indexes)].reset_index(drop=True)
train_indexes = dfTrain.index
dfValid = renamed_df.loc[renamed_df.index.isin(val_indexes)].reset_index(drop=True)
dfTrain = pd.concat([dfTrain, dfValid]).reset_index(drop=True)
val_indexes = dfTrain[-len(dfValid):].index

# dfTrain = renamed_df.loc[(renamed_df.index.isin(train_indexes)) | (renamed_df.index.isin(val_indexes))].reset_index(drop=True)
# dfValid = renamed_df.loc[renamed_df.index.isin(val_indexes)].reset_index(drop=True)
dfTest = renamed_df.loc[renamed_df.index.isin(test_indexes)].reset_index(drop=True)
df = df.loc[df.index.isin(test_indexes)].reset_index(drop=True)

training_generator, validation_generator, max_timesteps, num_features, X_test, y_test = prepare_data_for_lstm(dfTrain, dfTest, train_indexes, val_indexes, mode, mean_events_per_case, experiment_folder)

start_time = time.time()
if not exists(join(f'../{experiment_folder}', 'model', f"model_lstm.json")):
    model = train_model_generator(training_generator, validation_generator, max_timesteps, num_features, kpi_type, kpi_in_column,
                                    experiment_folder, args, epochs, n_layers=2, n_neurons=100)
    print(f"Time for training: {str(timedelta(seconds=round(time.time() - start_time)))}")
else:
    model = model_from_json(open(join(f'../{experiment_folder}', 'model', f"model_lstm.json")).read())
    model.load_weights(join(f'../{experiment_folder}', 'model', f"model_lstm_weights_best.h5"))

print("Testing model")
y_pred = model.predict(X_test)
y_pred = np.squeeze(y_pred)

if kpi_type == "Categorical":
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    y_pred = y_pred.astype("int")

common.write_test_set_results(df, y_pred, y_test, kpi_type, args.kpi, experiment_folder)

# save results and remove previous saved model
fromDirectory = f"../{experiment_folder}"
toDirectory = f'../experiments/{args.experiment_name}'
shutil.copytree(fromDirectory, toDirectory)