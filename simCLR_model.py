import tensorflow as tf

from graph_embeddings import *
from loss_functions import SimCLRNTXentLoss


class SimCLRModel(tf.keras.Model):
    def __init__(
            self,
            n_features,
            embedding_dim,
            output_dim,
            f_r_units, 
            f_o_units,
            phi_C_units,
            proj_units,
            **kwargs
    ):
        super().__init__(**kwargs)
        
        # Store configurations
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.f_r_units = f_r_units
        self.f_o_units = f_o_units
        self.phi_C_units = phi_C_units
        self.proj_units = proj_units

        # Create models
        self.encoder = self._create_encoder_model(
            self.n_features,
            self.embedding_dim,
            self.f_r_units,
            self.f_o_units,
            self.phi_C_units
        )
        
        self.projection_head = self._create_projection_head(
            self.embedding_dim,
            self.output_dim,
            self.proj_units
        )
    
    def build(self, input_shape):        
        # Two inputs [view1, view2], each with shape (batch_size, n_features, None)
        self.encoder.build(input_shape[0])
        self.projection_head.build((None, self.embedding_dim))

        super().build(input_shape)

    def call(self, inputs):
        # Unpack the two views
        view1, view2 = inputs
        
        # Get embeddings for both views using the shared encoder.
        # Input dimension (n_features, number of particles)
        # Output dimension (embedding_dim,)
        z1 = self.encoder(view1)
        z2 = self.encoder(view2)
        
        # Apply projection head to both embeddings
        # Input dimension (embedding_dim,)
        # Output dimension (output_dim,)
        p1 = self.projection_head(z1)
        p2 = self.projection_head(z2)
        
        return tf.stack([p1, p2], axis=1)
    
    def _create_encoder_model(
            self,
            n_features,
            embedding_dim,
            f_r_units, 
            f_o_units,
            phi_C_units
    ):
        """
        Encodes each sample into an embedding with length EMBEDDING_DIM
        """
        input = tf.keras.layers.Input(shape=(n_features, None), dtype=tf.float32, name='encoder_input')
        
        # Reduce graph to vector embeddings
        O_Bar = GraphEmbeddings(
            f_r_units=f_r_units,
            f_o_units=f_o_units
        )(input)
        
        # Trainable function phi_C to transform GNN output to final embeddings
        phi_C = tf.keras.Sequential([
            layer
            for units in phi_C_units
            for layer in [
                tf.keras.layers.Dense(units),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ]
        ])(O_Bar)
        
        dense = tf.keras.layers.Dense(units=embedding_dim)(phi_C)
        output = tf.keras.layers.BatchNormalization()(dense)

        return tf.keras.Model(inputs=input, outputs=output, name='encoder')


    def _create_projection_head(
            self,
            embedding_dim,
            output_dim,
            proj_units,
    ):
        """
        Projection head to be applied to embeddings. Shown in simCLR paper to improve performance.
        """
        proj_input = tf.keras.layers.Input(shape=(embedding_dim,), dtype=tf.float32, name='proj_input')
        
        proj_function = tf.keras.Sequential([
            layer
            for units in proj_units
            for layer in [
                tf.keras.layers.Dense(units),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ]
        ])(proj_input)
        
        dense = tf.keras.layers.Dense(units=output_dim)(proj_function)
        proj_output = tf.keras.layers.BatchNormalization()(dense)
        
        projection_model = tf.keras.Model(inputs=proj_input, outputs=proj_output, name='projection_head')
        
        return projection_model


def create_simclr_model(
        n_features,
        embedding_dim,
        output_dim,
        f_r_units=(96, 64, 32), 
        f_o_units=(128, 64, 32),
        phi_C_units=(128,),
        proj_units=(24,),
        temp=0.1,
        learning_rate=5e-4,
):
    
    # Create and compile model
    model = SimCLRModel(
        n_features,
        embedding_dim,
        output_dim,
        f_r_units, 
        f_o_units,
        phi_C_units,
        proj_units,
    )

    loss_fn = SimCLRNTXentLoss(temp)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn
    )
    
    return model
