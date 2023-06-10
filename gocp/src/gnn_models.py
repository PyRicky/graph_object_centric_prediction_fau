
import tensorflow as tf
from spektral.layers import GlobalSumPool, ECCConv, GATConv, GlobalAvgPool, GatedGraphConv, GINConv, TopKPool, GCSConv, GlobalAttentionPool, GCNConv
from spektral.models import GeneralGNN
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Dropout


class GCSModel():
    def __init__(self, n_node_features, n_out):
        self.model = self.create(n_node_features, n_out)

    def create(self, n_node_features, n_out):
        X_in = Input(shape=(n_node_features,), name="X_in")
        A_in = Input(shape=(None,), sparse=True)
        I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

        X_1 = GCSConv(32, activation="relu")([X_in, A_in])
        X_1, A_1, I_1 = TopKPool(ratio=0.5)([X_1, A_in, I_in])
        X_2 = GCSConv(32, activation="relu")([X_1, A_1])
        X_2, A_2, I_2 = TopKPool(ratio=0.5)([X_2, A_1, I_1])
        X_3 = GCSConv(32, activation="relu")([X_2, A_2])
        X_3 = GlobalAvgPool()([X_3, I_2])
        output = Dense(n_out, activation="softmax")(X_3)

        model = Model(inputs=[X_in, A_in, I_in], outputs=output)
        print(model.summary())
        return model

    def get_model(self):
        return self.model


class ECC():

    def __init__(self, n_node_features, n_edge_features, n_out):
        self.model = self.create(n_node_features, n_edge_features, n_out)

    def create(self, n_node_features, n_edge_features, n_out):
        X_in = Input(shape=(n_node_features,), name="X_in")
        A_in = Input(shape=(None,), sparse=True)
        E_in = Input(shape=(n_edge_features,), name="E_in")
        I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

        X_1 = ECCConv(channels=100)([X_in, A_in, E_in])
        X_2 = GlobalSumPool()([X_1, I_in])

        output = Dense(n_out, activation="softmax")(X_2)

        return Model(inputs=[X_in, A_in, E_in, I_in], outputs=output)

    def get_model(self):
        return self.model


class GCNModel():

    def __init__(self, n_node_features, n_out, kpi, kpi_type):
        self.kpi = kpi
        self.kpi_type = kpi_type
        self.model = self.create(n_node_features, n_out)

    def create(self, n_node_features, n_out):

        X_in = Input(shape=(n_node_features,), name="X_in")
        A_in = Input(shape=(None,), sparse=True)
        I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

        # Venugopal et al. (2021)
        # A Comparison of Deep-Learning Methods for Analysing and Predicting Business Processes
        X_1 = GCNConv(1)([X_in, A_in])  # one channel
        X_1 = Dropout(0.5)(X_1)
        X_1 = Dense(256, activation="tanh")(X_1)  # number nodes x 256
        X_1 = Dense(256, activation="tanh")(X_1)  # 256 x 256
        X_1 = Dropout(0.5)(X_1)
        X_1 = GlobalAvgPool()([X_1, I_in])

        if self.kpi == "next_activity":
            output = Dense(n_out, activation="softmax")(X_1)  # 256 x number nodes
        elif self.kpi_type == "Numerical":
            output = Dense(n_out)(X_1)
        else:
            output = Dense(n_out, activation="sigmoid")(X_1)

        model = Model(inputs=[X_in, A_in, I_in], outputs=output)
        print(model.summary())
        return model

    def get_model(self):
        return self.model


class GATConvModel():

    def __init__(self, n_node_features, n_out):
        self.model = self.create(n_node_features, n_out)

    def create(self, n_node_features, n_out):

        X_in = Input(shape=(n_node_features,), name="X_in")
        A_in = Input(shape=(None,), sparse=True)
        I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

        X_1 = GATConv(channels=64, attn_heads=4)([X_in, A_in])
        X_2 = GlobalSumPool()([X_1, I_in])
        output = Dense(n_out, activation="softmax")(X_2)

        model = Model(inputs=[X_in, A_in, I_in], outputs=output)
        print(model.summary())
        return model

    def get_model(self):
        return self.model


class GatedGraphConvModel():

    def __init__(self, n_node_features, n_edge_features, n_out, kpi, kpi_type):
        self.kpi = kpi
        self.kpi_type = kpi_type
        self.model = self.create(n_node_features, n_edge_features, n_out)

    def create(self, n_node_features, n_edge_features, n_out):
        X_in = Input(shape=(n_node_features,), name="X")
        A_in = Input(shape=(None,), sparse=True, name="A")
        E_in = Input(shape=(n_edge_features,), name="E")
        I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

        X_1 = GatedGraphConv(channels=n_node_features, n_layers=2, activation="tanh")([X_in, A_in, E_in])
        # X_1 = GatedGraphConv(channels=n_node_features, n_layers=2, activation="tanh")([X_in, A_in])

        X_2 = GlobalAttentionPool(100)([X_1, I_in])  # 80?

        # X_2 = Dense(300, activation='tanh')(X_2)
        # X_2 = Dropout(0.5)(X_2)
        # X_2 = Dense(200, activation='tanh')(X_2)
        # X_2 = Dropout(0.5)(X_2)
        # X_2 = Dense(100, activation='tanh')(X_2)
        # X_2 = Dropout(0.5)(X_2)

        if self.kpi == "next_activity":
            output = Dense(n_out, activation="softmax")(X_2)
        elif self.kpi_type == "Numerical":
            output = Dense(n_out)(X_2)
        else:
            output = Dense(n_out, activation="sigmoid")(X_2)

        model = Model(inputs=[X_in, A_in, E_in, I_in], outputs=output)
        # model = Model(inputs=[X_in, A_in, I_in], outputs=output)
        print(model.summary())
        return model

    def get_model(self):
        return self.model


class generalGNN():

    def __init__(self, n_out):
        self.model = self.create(n_out)

    def create(self, n_out):
        return GeneralGNN(n_out, activation="softmax")  # https://graphneural.network/models/#generalgnn

    def get_model(self):
        return self.model


class GIN0Model():

    def __init__(self, n_node_features, n_out):
        self.model = self.create(n_node_features, n_out)

    def create(self, n_node_features, n_out):
        channels = n_node_features * 100  # Hidden units
        layers = 3  # GIN layers
        return GIN0(channels, layers, n_out)

    def get_model(self):
        return self.model


class GIN0(Model):

    def __init__(self, channels, n_layers, n_out):
        super().__init__()
        self.conv1 = GINConv(channels, epsilon=0, mlp_hidden=[channels, channels])
        self.convs = []
        for _ in range(1, n_layers):
            self.convs.append(
                GINConv(channels, epsilon=0, mlp_hidden=[channels, channels])
            )
        self.pool = GlobalAvgPool()
        self.dense1 = Dense(channels, activation="relu")
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(n_out, activation="softmax")

    def call(self, inputs):
        x, a, i = inputs
        x = self.conv1([x, a])
        for conv in self.convs:
            x = conv([x, a])
        x = self.pool([x, i])
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)
