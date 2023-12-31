import tensorflow as tf


class MLP(tf.keras.Model):
    def __init__(self, num_hidden, dim_hidden, dim_out=None, dropout=0.0, batch_norm=True, name="MLP"):
        super().__init__(name=name)

        self.dim_out = dim_out
        self.blocks = tf.keras.Sequential(name="MLP")
        for _ in range(num_hidden - 1):
            self.blocks.add(tf.keras.layers.Dense(dim_hidden))

            if batch_norm:
                self.blocks.add(tf.keras.layers.BatchNormalization())

            self.blocks.add(tf.keras.layers.ReLU())
            self.blocks.add(tf.keras.layers.Dropout(dropout))

        if dim_out:
            self.blocks.add(tf.keras.layers.Dense(dim_out))
        else:
            self.blocks.add(tf.keras.layers.Dense(dim_hidden))

    def call(self, inputs, training=None):
        return self.blocks(inputs, training=training)


class FeatureSelection(tf.keras.Model):
    def __init__(
        self,
        dim_input,
        dim_embedding,
        dim_gate=None,
        num_hidden=1,
        dim_hidden=64,
        conditional_indices1=None,
        conditional_indices2=None,
        dropout=0.0,
    ):
        super().__init__()

        if not dim_gate:
            dim_gate = dim_input

        self.dim_embedding = dim_embedding
        self.dim_input = dim_input

        if conditional_indices1:
            self.conditional_indices1 = tf.constant(conditional_indices1, dtype=tf.int32)
            dim_out_1 = len(conditional_indices1) * dim_embedding
        else:
            self.conditional_indices1 = None
            self.gate_1_bias = self.add_weight(
                shape=(1, dim_gate), initializer="ones", trainable=True, name="gate_1_bias"
            )
            dim_out_1 = dim_input * dim_embedding

        self.gate_1 = MLP(
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=dim_out_1,
            dropout=dropout,
            batch_norm=False,
            name="feature_selection_gate_1",
        )

        if conditional_indices2:
            self.conditional_indices2 = tf.constant(conditional_indices2, dtype=tf.int32)
            dim_out_2 = len(conditional_indices2) * dim_embedding
        else:
            self.conditional_indices2 = None
            self.gate_2_bias = self.add_weight(
                shape=(1, dim_gate), initializer="ones", trainable=True, name="gate_2_bias"
            )
            dim_out_2 = dim_input * dim_embedding

        self.gate_2 = MLP(
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=dim_out_2,
            dropout=dropout,
            batch_norm=False,
            name="feature_selection_gate_2",
        )

    def call(self, embeddings, training=None):
        # embeddings is of shape (batch_size, dim_input, dim_embedding)

        if self.conditional_indices1 is not None:
            x1 = tf.gather(embeddings, self.conditional_indices1, axis=1)  # (bs, num_cond_inds1, dim_embedding)
            # (bs, num_cond_ind1 * dim_embedding)
            x1 = tf.reshape(x1, (-1, len(self.conditional_indices1) * self.dim_embedding))
            gate_score_1 = self.gate_1(x1, training=training)  # (1, num_cond_inds1 * dim_embedding)
            out_1 = 2.0 * tf.nn.sigmoid(gate_score_1) * x1  # (bs, num_cond_inds1 * dim_embedding)
        else:
            embeddings1 = tf.reshape(embeddings, (-1, self.dim_input * self.dim_embedding))
            x1 = self.gate_1_bias  # (1, dim_feature)
            gate_score_1 = self.gate_1(x1, training=training)  # (1, dim_feature)
            out_1 = 2.0 * tf.nn.sigmoid(gate_score_1) * embeddings1  # (bs, dim_feature)

        if self.conditional_indices2 is not None:
            x2 = tf.gather(embeddings, self.conditional_indices2, axis=1)  # (bs, num_cond_inds2, dim_embedding)
            # (bs, num_cond_inds2 * dim_embedding)
            x2 = tf.reshape(x2, (-1, len(self.conditional_indices2) * self.dim_embedding))
            gate_score_2 = self.gate_2(x2, training=training)  # (1, num_cond_inds2 * dim_embedding)
            out_2 = 2.0 * tf.nn.sigmoid(gate_score_2) * x2  # (bs, num_cond_inds2 * dim_embedding)
        else:
            embeddings2 = tf.reshape(embeddings, (-1, self.dim_input * self.dim_embedding))
            x2 = self.gate_2_bias  # (1, dim_feature)
            gate_score_2 = self.gate_2(x2, training=training)  # (1, dim_feature)
            out_2 = 2.0 * tf.nn.sigmoid(gate_score_2) * embeddings2  # (bs, dim_feature)

        return out_1, out_2  # (bs, dim_feature), (bs, dim_feature)


class Aggregation(tf.keras.Model):
    def __init__(self, dim_latent_1, dim_latent_2, num_heads=1, name="aggregation"):
        super().__init__(name=name)

        self.num_heads = num_heads
        self.dim_head_1 = dim_latent_1 // num_heads
        self.dim_head_2 = dim_latent_2 // num_heads

        self.w_1 = self.add_weight(
            shape=(self.dim_head_1, num_heads, 1), initializer="glorot_uniform", trainable=True, name="w1"
        )
        self.w_2 = self.add_weight(
            shape=(self.dim_head_2, num_heads, 1), initializer="glorot_uniform", trainable=True, name="w2"
        )
        self.bias = self.add_weight(shape=(1, num_heads, 1), initializer="zeros", trainable=True, name="bias")
        self.w_12 = self.add_weight(
            shape=(num_heads, self.dim_head_1, self.dim_head_2, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="w12",
        )

    def call(self, latent_1, latent_2):
        # bilinear aggregation of the two latent representations
        # y = b + w_1.T o_1 + w_2.T o_2 + o_1.T W_3 o_2
        # first_order = self.w_1(latent_1) + self.w_2(latent_2)  # (bs, 1)

        latent_1 = tf.reshape(latent_1, (-1, self.num_heads, self.dim_head_1))  # (bs, num_heads, dim_head_1)
        latent_2 = tf.reshape(latent_2, (-1, self.num_heads, self.dim_head_2))  # (bs, num_heads, dim_head_2)

        first_order = tf.einsum("bhi,iho->bho", latent_1, self.w_1)  # (bs, num_heads, 1)
        first_order += tf.einsum("bhi,iho->bho", latent_2, self.w_2)  # (bs, num_heads, 1)

        second_order = tf.einsum("bhi,hijo,bhj->bho", latent_1, self.w_12, latent_2)  # (bs, num_heads, 1)

        out = tf.reduce_sum(first_order + second_order + self.bias, 1)  # (bs, 1)

        return out


class FinalMLP(tf.keras.Model):
    def __init__(
        self,
        dim_input,
        num_embedding,
        dim_embedding=8,
        embedding_l2=0.0,
        dim_hidden_fs=64,
        num_hidden_1=2,
        dim_hidden_1=64,
        conditional_indices_1=None,
        num_hidden_2=2,
        dim_hidden_2=64,
        conditional_indices_2=None,
        num_heads=1,
        dropout=0.0,
        name="FinalMLP",
    ):
        super().__init__(name=name)

        self.dim_input = dim_input
        self.dim_embedding = dim_embedding

        # embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=num_embedding,
            output_dim=dim_embedding,
            input_length=dim_input,
            embeddings_regularizer=tf.keras.regularizers.l2(l2=embedding_l2),
            name="embedding",
        )

        # feature selection layer that projects a learnable vector to the flatened embedded feature space
        self.feature_selection = FeatureSelection(
            dim_input=dim_input,
            dim_embedding=dim_embedding,
            dim_gate=dim_input,
            dim_hidden=dim_hidden_fs,
            conditional_indices1=conditional_indices_1,
            conditional_indices2=conditional_indices_2,
            dropout=dropout,
        )

        # branch 1
        self.interaction_1 = MLP(
            num_hidden=num_hidden_1,
            dim_hidden=dim_hidden_1,
            dropout=dropout,
            name="MLP_1",
        )
        # branch 2
        self.interaction_2 = MLP(
            num_hidden=num_hidden_2,
            dim_hidden=dim_hidden_2,
            dropout=dropout,
            name="MLP_2",
        )

        # final aggregation layer
        self.aggregation = Aggregation(dim_latent_1=dim_hidden_1, dim_latent_2=dim_hidden_2, num_heads=num_heads)

        self.build(input_shape=(None, dim_input))

    def call(self, inputs, training=None):
        # (batch_size, dim_input, embedding_dim)
        embeddings = self.embedding(inputs, training=training)

        # weight features of the two streams using a gating mechanism
        emb_1, emb_2 = self.feature_selection(embeddings, training=training)

        # get interactions from the two branches
        latent_1 = self.interaction_1(emb_1, training=training)
        latent_2 = self.interaction_2(emb_2, training=training)

        # merge the representations using an aggregation scheme
        logits = self.aggregation(latent_1, latent_2)
        outputs = tf.nn.sigmoid(logits)

        return outputs
