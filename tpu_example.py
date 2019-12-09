import tensorflow as tf
import tensorflow_transform as tft

import numpy as np
from absl import app, flags
from datetime import datetime

from util import print_fn

FLAGS = flags.FLAGS
# Cloud TPU Cluster Resolver flags
flags.DEFINE_string(
    "tpu", default="node-1",
    help="Cloud TPU name" )
flags.DEFINE_string(
    "tpu_zone", default="us-central1-b",
    help="GCE zone where the Cloud TPU is located in" )
flags.DEFINE_string(
    "gcp_project", default="ox-lab-junbo-park",
    help="Project name of the Cloud TPU-enabled project" )

flags.DEFINE_integer("train_steps", 10, "Total number of training steps")
flags.DEFINE_integer("eval_steps", 0, "Total number of evaluation steps")
flags.DEFINE_float("learning_rate", 0.05, "Learning rate")
flags.DEFINE_integer("embedding_dim", 5, "dimensions for embeddings")
flags.DEFINE_integer("iterations", 10, "Numer of iterations per TPU training loop")
flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips)")

def input_fn(params):
    batch_size_per_core = 9
    dense_shape = (batch_size_per_core, 1)
    indices = ((0,0), (3,0), (4,0), (5,0), (7,0), (8,0) )
    values  = (0, 2, 1, 0, 1, 0)
    # values  = ("car", "ship", "flower", "car", "flower", "car")
    inputs  = tf.sparse.SparseTensor(
        indices=indices, values=values, dense_shape=dense_shape)

    # Manually set labels
    labels = np.zeros(dense_shape, dtype=np.float32)
    labels[0, 0] = 1
    labels[5, 0] = 1
    labels[8, 0] = 1

    labels = tf.cast(labels, tf.int32)

    data = tf.data.Dataset.from_tensors(({'objects': inputs}, labels))
    return data.repeat()

def create_embedding_config_spec_and_model_fn():
    num_buckets = 3
    sparse_id_column = tf.feature_column.categorical_column_with_identity(
        key='objects', num_buckets=num_buckets)

    initializer = tf.random_normal_initializer(0., 0.01)
    feature_columns = [
        tf.compat.v1.tpu.experimental.embedding_column(
            categorical_column=sparse_id_column,
            dimension=FLAGS.embedding_dim,
            combiner=None,
            initializer=initializer
        )
    ]
    
    optimization_parameters = tf.compat.v1.tpu.experimental.AdamParameters(
        learning_rate=0.5
    )
    embedding_config_spec = tf.compat.v1.estimator.tpu.experimental.EmbeddingConfigSpec(
                    feature_columns, optimization_parameters)

    def model_fn(features, labels, mode, params):
        
        print_fn("features", features)
        
        input_layer = tf.keras.layers.DenseFeatures(feature_columns)(features)
        print_fn("input_layer", input_layer)

        y = tf.layers.Dense(3, activation="relu")(input_layer)
        logits = tf.layers.Dense(2)(y)

        print_fn("labels", labels)
        print_fn("logits", logits)

        # loss = tf.losses.mean_squared_error(labels, logits)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step()
        )
        
        return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, train_op=train_op, loss=loss)
    
    return embedding_config_spec, model_fn

def main(argv):
    del argv

    tf.logging.set_verbosity(tf.logging.INFO)

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu,
        zone=FLAGS.tpu_zone,
        project=FLAGS.gcp_project
    )

    tpu_config = tf.contrib.tpu.TPUConfig(
        FLAGS.iterations, FLAGS.num_shards,
        per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    )
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=FLAGS.model_dir,
        tpu_config=tpu_config
    )

    embedding_config_spec, model_fn = (create_embedding_config_spec_and_model_fn())
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=True,
        train_batch_size=72,
        config=run_config,
        embedding_config_spec=embedding_config_spec
    )

    estimator.train(input_fn=input_fn, max_steps=FLAGS.train_steps)

if __name__ == "__main__":
    app.run(main)
