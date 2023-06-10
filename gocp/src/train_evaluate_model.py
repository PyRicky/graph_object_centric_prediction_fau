from spektral.data import DisjointLoader
from gocp.src.gnn_models import ECC, GatedGraphConvModel, GATConvModel, \
    generalGNN, GIN0Model, GCSModel, GCNModel
from gocp.src.preprocessing.common import write_and_plot_test_error
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle, json
from gocp.src.preprocessing import common


class Model():

    def __init__(self, args, n_node_features, n_edge_features, n_out, kpi_type, kpi_in_column, batch_size, ohe_mapping, activity_id_name_mapping, learning_rate):
        self.args = args
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_out = n_out
        self.kpi_type = kpi_type
        self.kpi_in_column = kpi_in_column
        self.batch_size = batch_size
        self.ohe_mapping = ohe_mapping
        self.activity_id_name_mapping = activity_id_name_mapping
        self.opt = Adam(learning_rate=learning_rate)
        self.loss_fn = self.calculate_loss_fn()
        self.model = self.build_model()


    def calculate_loss_fn(self):
        if self.args.kpi == "activity" or (self.kpi_in_column and self.kpi_type == "Categorical"):
            loss_fn = BinaryCrossentropy()
        elif self.args.kpi == "next_activity":
            loss_fn = CategoricalCrossentropy()
        else:
            loss_fn = MeanSquaredError()
        return loss_fn


    def build_model(self):
        # BUILD MODEL
        if self.args.architecture == 'ECC':
            model = ECC(self.n_node_features, self.n_edge_features, self.n_out).get_model()
        elif self.args.architecture == 'GAT':
            model = GATConvModel(self.n_node_features, self.n_out).get_model()
        elif self.args.architecture == 'GCS':
            model = GCSModel(self.n_node_features, self.n_out).get_model()
        elif self.args.architecture == 'GatedGraph':
            model = GatedGraphConvModel(self.n_node_features, self.n_edge_features, self.n_out, self.args.kpi, self.kpi_type).get_model()
        elif self.args.architecture == 'GNN':
            model = generalGNN(self.n_out).get_model()
        elif self.args.architecture == 'GIN0':
            model = GIN0Model(self.n_node_features, self.n_out).get_model()
        elif self.args.architecture == 'GCN':
            model = GCNModel(self.n_node_features, self.n_out, self.args.kpi, self.kpi_type).get_model()

        else:
            raise ValueError("Not a defined Model")
        return model
        

    # FIT MODEL (PER BATCH)
    # @tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
    def train_step(self, inputs, target):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(target, predictions)
            loss += sum(self.model.losses)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def evaluate_model(self, dataset_batch, indexes, batch_size, evaluation_type, df, experiment_folder, compare=False):
        y_pred = []
        y_pred_probs = []
        y_true = []
        output = []
        
        batch_count = 0
        # access all the graphs, and then fit one batch at a time
        for i in range(0, len(indexes)):
            if evaluation_type == "test":
                with open(f'../graph_chunks/test_batch_{indexes[i]}.pickle', 'rb') as handle:
                    dataset_batch.graphs.append(pickle.load(handle))
            else:
                with open(f'../graph_chunks/valid_batch_{indexes[i]}.pickle', 'rb') as handle:
                    dataset_batch.graphs.append(pickle.load(handle))
            batch_count += 1
            # if we have enough elements for a batch or it is the last step and the number of samples is not divisible by the batch size
            if (batch_count == batch_size) or (((i+1) == len(indexes)) and (batch_count != 0)):
                loader = DisjointLoader(dataset_batch, batch_size=batch_size, shuffle=False)
                inputs, target = loader.__next__()

                pred = self.model(inputs, training=False)
                outs = (self.loss_fn(target, pred),)
                output.append(outs)
                
                dataset_batch.graphs = []
                batch_count = 0

                # at the end of the batch, append y_true and y_pred also for validation set once the model is trained in order to obtain the precise scores over the validation set
                if evaluation_type == "valid" and compare is True:
                    y_true.extend(target.T.ravel().tolist())
                    pred_proba = np.array(pred)
                    if self.args.kpi == "activity" or (self.kpi_in_column and self.kpi_type == "Categorical"):
                        pred_proba[pred_proba > 0.5] = 1
                        pred_proba[pred_proba < 0.5] = 0
                        y_pred.extend(pred_proba.astype('int').T.ravel().tolist())
                    else:
                        y_pred.extend(pred_proba.T.ravel().tolist())


                # at the end of the batch, extend predictions and targets for test set 
                if evaluation_type == "test":
                    if self.args.kpi == "next_activity":
                        for row in target:
                            y_true.append(self.ohe_mapping[str(row.astype(int).tolist())])
                    else:
                        y_true.extend(target.T.ravel().tolist())

                    pred_proba = np.array(pred)

                    if self.args.kpi == "next_activity":
                        for idx in range(0, len(pred_proba)):  # get probability distribution of prediction of the model
                            y_pred_probs.append(list(pred_proba[idx, :]))

                        y_pred_labels = np.array(math_ops.argmax(pred, axis=-1))  # get prediction of the model
                        for y_pred_label in y_pred_labels:
                            y_pred.append(self.activity_id_name_mapping[y_pred_label])

                    elif self.args.kpi == "activity" or (self.kpi_in_column and self.kpi_type == "Categorical"):
                        pred_proba[pred_proba > 0.5] = 1
                        pred_proba[pred_proba < 0.5] = 0
                        y_pred.extend(pred_proba.astype('int').T.ravel().tolist())
                    
                    else:
                        y_pred.extend(pred_proba.T.ravel().tolist())

        # at the end, compute overall statistics on the test set 
        if evaluation_type == "test":
            common.write_test_set_results(df, y_pred, y_true, self.kpi_type, self.args.kpi, experiment_folder)

        if evaluation_type == "valid" and compare is True:
            # to be called once the model is trained in order to obtain the scores over the validation set in order to compute statistics per prefix (to be compared with Catboost later)
            y_pred = pd.Series(y_pred).rename("Prediction")
            y_true = pd.Series(y_true).rename('TEST')

            df_results_valid = pd.concat([df.loc[df.index.isin(indexes), "Case ID"].reset_index(drop=True), y_pred.reset_index(drop=True), y_true.reset_index(drop=True)], axis=1)
            df_results_valid["Prefix"] = df_results_valid.groupby("Case ID").cumcount()+1
            df_results_valid.to_csv(f"../{experiment_folder}/result/results_valid.csv", index=False)

        return np.mean(output, 0)