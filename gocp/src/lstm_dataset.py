import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from os.path import exists, join
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, LSTM, Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.models import model_from_json
from keras.preprocessing import sequence
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import keras


def one_hot_encoding(df):
	for column in df.columns:
		if df[column].dtype == 'object' and column != "case:concept:name":
			# One hot encoding - eventual categorical nans will be ignored
			one_hot = pd.get_dummies(df[column].astype('str'), prefix=column, prefix_sep='=')
			one_hot.drop(list(one_hot.filter(regex='nan|missing')), axis=1, inplace=True)
			print("Encoded column:{} - Different keys: {}".format(column, one_hot.shape[1]))
			# Drop column as it is now encoded
			df = df.drop(column, axis=1)
			# Join the encoded df
			df = df.join(one_hot)
	print("Categorical columns encoded")
	return df


def normalize_data(df):
    case_column = df["case:concept:name"].copy()
    del df[case_column.name]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.values)
    df = pd.DataFrame(scaled, columns=df.columns)
    df.insert(0, case_column.name, case_column)
    print("Normalized Data")
    return df


def generate_sequences_for_lstm(dataset, mode, experiment_folder, type):
    """
    Input: dataset
    Output: sequences of partial traces and remaining times to feed LSTM
    1 list of traces, each containing a list of events, each containing a list of attributes (features)
    """
    # if we are in real test we just have to generate the single sequence of all events, not 1 sequence per event
    data = []
    trace = []
    caseID = dataset[0][0]
    trace.append(dataset[0][1:].tolist())
    for line in dataset[1:, :]:
        case = line[0]
        if case == caseID:
            trace.append(line[1:].tolist())
        else:
            caseID = case
            if mode == "train":
                for j in range(1, len(trace) + 1):
                    data.append(trace[:j])
            else:
                data.append(trace[:len(trace)])
            trace = []
            trace.append(line[1:].tolist())
    # last case
    if mode == "train":
        for i in range(1, len(trace) + 1):
            data.append(trace[:i])
    else:
        data.append(trace[:len(trace)])
    #save list to file
    if type == "train":
        np.save(f"../{experiment_folder}/model/train_sequences.npy", np.array(data))
    else:
        np.save(f"../{experiment_folder}/model/test_sequences.npy", np.array(data))
    return data



class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, list_indexes, targets, max_timesteps, experiment_folder, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.targets = targets
        self.list_indexes = list_indexes
        self.max_timesteps = max_timesteps
        self.shuffle = shuffle
        self.on_epoch_end()
        self.data = np.load(f'../{experiment_folder}/model/train_sequences.npy', allow_pickle=True).tolist()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_indexes) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Take indexes of the current batch
        second_index = (index + 1) * self.batch_size
        if second_index < len(self.list_indexes):
            batch_indexes = self.list_indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            #last iteration
            batch_indexes = self.list_indexes[index*self.batch_size:]

        # Generate data
        X, y = self.__data_generation(batch_indexes)
        return X, y

    def on_epoch_end(self):
        """Shuffle indexes after each epoch"""
        if self.shuffle == True:
            np.random.shuffle(self.list_indexes)


    def __data_generation(self, list_indexes_batch):
        """Generates data containing batch_size samples"""
        X = []
        y = []
        for i in list_indexes_batch:
            # take only up to n timesteps for performance reasons
            X.append(self.data[i][:self.max_timesteps])
            # access pathtime target referring to index (row) i
            y.append(self.targets[i])

        X = sequence.pad_sequences(X, maxlen=self.max_timesteps, dtype="float32")
        # TODO: the code wass breaking because it is trying to do a .shape on a list - is this the correct solution?
        return np.array(X), np.array(y)


def prepare_data_for_lstm(dfTrain, dfTest, train_indexes, val_indexes, mode, mean_events_per_case, experiment_folder):
    y_train = dfTrain.iloc[:, -1]
    y_test = dfTest.iloc[:, -1]
    X_train = dfTrain.iloc[:, :-1]
    X_test = dfTest.iloc[:, :-1]

    # test columns should be the same as train
    columns_not_in_test = [x for x in X_train.columns if x not in X_test.columns]
    df2 = pd.DataFrame(columns=columns_not_in_test)
    # enrich test df with missing columns seen in train
    X_test = pd.concat([X_test, df2], axis=1)
    # reorder columns as in train (otherwise the model crashes)
    X_test = X_test[X_train.columns]

    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)

    X_train = generate_sequences_for_lstm(X_train.values, mode, experiment_folder, type="train")
    X_test = generate_sequences_for_lstm(X_test.values, mode, experiment_folder, type="test")
    assert (len(y_train) == len(X_train))
    assert (len(y_test) == len(X_test))

    num_features = len(X_train[0][0])
    max_timesteps = dfTrain.groupby("case:concept:name").count().max().iloc[0]
    print(f"Max number of timesteps observed: {max_timesteps}")
    max_timesteps = max_timesteps if max_timesteps < mean_events_per_case else mean_events_per_case
    print(f"DEBUG: training shape ({len(X_train)},{max_timesteps},{num_features})")

    #save train data to be reloaded later in batches
    np.save(f'../{experiment_folder}/model/train_sequences.npy', X_train)

    # Parameters
    params = {'batch_size': 32, 'shuffle': True}

    # divide into train and validation cases (validation is 20%)
    index_partition_cases = {'train': train_indexes.to_list(), 'validation': val_indexes.to_list()}
    targets = dfTrain["y"].to_dict()

    # Generators
    training_generator = DataGenerator(index_partition_cases['train'], targets, max_timesteps, experiment_folder, **params)
    validation_generator = DataGenerator(index_partition_cases['validation'], targets, max_timesteps, experiment_folder, **params)

    for i in range(len(X_test)):
        X_test[i] = X_test[i][:max_timesteps]
    X_test = sequence.pad_sequences(X_test, maxlen=max_timesteps, dtype="float32")
    print(f"DEBUG: test shape ({len(X_test)},{max_timesteps},{len(X_test[0][0])})")

    return training_generator, validation_generator, max_timesteps, num_features, X_test, y_test


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def train_model_generator(training_generator, validation_generator, max_timesteps, num_features, kpi_type, kpi_in_column,
                            experiment_folder, args, num_epochs=200, n_neurons=100, n_layers=2):
    # create the model
    model = Sequential()
    if n_layers == 1:
        model.add(LSTM(n_neurons, implementation=2, input_shape=(max_timesteps, num_features),
                        recurrent_dropout=0.2))
        model.add(BatchNormalization())
    else:
        for i in range(n_layers - 1):
            model.add(LSTM(n_neurons, implementation=2, input_shape=(max_timesteps, num_features),
                            recurrent_dropout=0.2, return_sequences=True))
            model.add(BatchNormalization())
        model.add(LSTM(n_neurons, implementation=2, recurrent_dropout=0.2))
        model.add(BatchNormalization())

    # add output layer (both regression and activity prediction output one number)
    model.add(Dense(1))
    if kpi_type == 'Numerical':
        # compiling the model, creating the callbacks
        model.compile(loss='mae', optimizer='Nadam', metrics=['mean_squared_error', 'mae', 'mape'])
    else:
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy', 'binary_accuracy', f1])   

    print(model.summary())
    early_stopping = EarlyStopping(patience=25)
    model_checkpoint = ModelCheckpoint(join(f'../{experiment_folder}', 'model', f"model_weights_best.h5"),
        monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                    epsilon=0.0001, cooldown=0, min_lr=0)

    callbacks = [early_stopping, model_checkpoint, lr_reducer]

    # Train model on dataset use_multiprocessing=True, workers=6,
    history = model.fit_generator(generator=training_generator,
                        validation_data=validation_generator, epochs=num_epochs, callbacks=callbacks)

    history_frame = pd.DataFrame.from_dict(history.history, orient='index')
    epochs = history_frame.columns.to_list()
    train_loss = history_frame.loc["loss"].tolist()
    valid_loss = history_frame.loc["val_loss"].tolist()
    plt.plot(epochs, train_loss, label = "Train loss")
    plt.plot(epochs, valid_loss, label = "Valid loss")
    plt.xlabel("Epochs")
    if args.kpi == "activity" or (kpi_in_column and kpi_type == "Categorical"):
        plt.ylabel("Binary Crossentropy")
    elif args.kpi == "next_activity":
        plt.ylabel("Categorical Crossentropy")
    else:
        plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.savefig(f"../{experiment_folder}/result/training_loss.png", dpi=300, bbox_inches="tight")

    # saving model shape to file
    model_json = model.to_json()
    with open(join(f'../{experiment_folder}', 'model', f"model.json"), "w") as json_file:
        json_file.write(model_json)
    print("Created model and saved weights")
    return model