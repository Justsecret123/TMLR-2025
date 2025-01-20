#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:35:21 2023

@author: ibrahimserouis
"""


import functools
import gc
import itertools
import pickle
import traceback
import numpy as np
import tensorflow_gnn as tfgnn
import tensorflow as tf

# Set the possible values for the scene graph info
INFO = [True,False]
# Set the list of training flags
TRAIN_FLAGS = list(itertools.product(INFO,repeat=7))
# Set the global batch size 
GLOBAL_BATCH_SIZE = 32
# Load the graph schema
GRAPH_SCHEMA = tfgnn.read_schema("mg_schema_FULL.pbtxt")
# Set the paience factor 
PATIENCE_FACTOR = 40
# Set the number of graph udpates 
G_UPDATES = 1
# Set the random seed 
tf.keras.utils.set_random_seed(123)


def load_dataset():
    """_summary_
    """

    # Load the data
    dataset = tf.data.TFRecordDataset("./tfrecords/mg_val_binary_full_completed.tfrecord")
    # Create a graph spec
    graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(GRAPH_SCHEMA)
    # Map the dataset to the schema
    dataset = dataset.map(lambda serialized: tfgnn.parse_single_example(graph_tensor_spec, serialized))
    # Sample data
    # Loop through the input
    for i, graph in enumerate(dataset.take(2)):
        # Display the sample
        print(f"Input {i}: {graph.node_sets['Interaction'].features}")

    # Get the class names
    class_names = set()

    # Get the data
    with open("class_names_2.pkl", "rb") as file:
        # Load the data
        class_names = pickle.load(file)
    
    global num_classes

    num_classes = len(class_names)
    print(f"Num classes : {num_classes}")
    
    return dataset, class_names

def setup_strategy():
    """_summary_
    """

    global strategy 

    try:
        tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Running on TPU ", tpu_resolver.cluster_spec().as_dict()["worker"])
    except ValueError:
        tpu_resolver = None
    if tpu_resolver:
        print("Using TPUStrategy")
        tf.config.experimental_connect_to_cluster(tpu_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
        strategy = tf.distribute.TPUStrategy(tpu_resolver)
        assert isinstance(strategy, tf.distribute.TPUStrategy)
    elif tf.config.list_physical_devices("GPU"):
        print(f"Using MirroredStrategy for GPUs")
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
        print(f"Using default strategy")
        print(f"Found {strategy.num_replicas_in_sync} replicas in sync")

    return strategy

def preprocess_node_features(node_set:tfgnn.NodeSet, node_set_name:str):
    """_summary_

    Args:
        node_set (tfgnn.NodeSet): _description_
        node_set_name (str): _description_
    """

    # Process untrainable id on Interaction
    # Generate Interactions embedding
    # Return the unprocessed summary
    if node_set_name=="Interaction":
        return {"summary": node_set["summary"]}
    
    # Scene
    if node_set_name=="Scene":
        return {
            "objectification": node_set["objectification"],
            "empty_state": tfgnn.keras.layers.MakeEmptyFeature()(node_set)}
    
    # Character
    # 3027 characters, generate 4000 bins
    if node_set_name=="Character":
        if train_flag[0]:
           return {"name": node_set["name"]}
        else:
           return {"empty_state": tfgnn.keras.layers.MakeEmptyFeature()(node_set)}
        
    # Relationship
    # 2122 relationship, generate 2500 bins
    if node_set_name=="Relationship":
        if train_flag[1]:
           return {"type": node_set["type"]}
        else:
           return {"empty_state": tfgnn.keras.layers.MakeEmptyFeature()(node_set)}
    
    # Reason
    if node_set_name=="Reason":
        return {"empty_state": tfgnn.keras.layers.MakeEmptyFeature()(node_set)}
    
    # Emotion
    # 34153 emotions, generate 35000 bins
    if node_set_name=="Emotion":
        if train_flag[2]:
           return {"desc": node_set["description"]}
        else:
           return {"empty_state": tfgnn.keras.layers.MakeEmptyFeature()(node_set)}

    # Attribute
    # 365 attribute names, generate 400*2 bins
    # 365 attribute values, generate 400*2 bins
    if node_set_name=="Attribute":
        if train_flag[3]:
           return {"value": node_set["value"]}
        else:
           return {"empty_state": tfgnn.keras.layers.MakeEmptyFeature()(node_set)}    
        
    # Place
    # 773 places, generate 1000 bins
    if node_set_name=="Place":
        if train_flag[4]:
           return {"desc": node_set["description"]}
        else:
           return {"empty_state": tfgnn.keras.layers.MakeEmptyFeature()(node_set)}    
        
    # Context
    # 1826 contexts, generate 2000 bins
    if node_set_name=="Context":
        if train_flag[5]:
           return {"desc": node_set["description"]}
        else: 
           return {"empty_state": tfgnn.keras.layers.MakeEmptyFeature()(node_set)} 
    
    # Speech
    if node_set_name=="Speech":
        if train_flag[6]:
           return {"transcript": node_set["transcript"]}
        else:
           return {"empty_state": tfgnn.keras.layers.MakeEmptyFeature()(node_set)} 
   # Frames
    #if node_set_name=="Frames":
    #    if train_flag[7]:
    #       return {"representation": node_set["representation"]}
    #    else:
    #       return {"empty_state": tfgnn.keras.layers.MakeEmptyFeature()(node_set)} 

def preprocess_edge_features(edge_set:tfgnn.EdgeSet, edge_set_name:str):
    """_summary_

    Args:
        edge_set (tfgnn.EdgeSet): _description_
        edge_set_name (str): _description_
    """

    # 5000 bins
    if edge_set_name=="involves" or edge_set_name=="linked_to":
        return {"hashed_role": tf.keras.layers.Hashing(5_0000)(edge_set["role"])}

def drop_all_features(_, **unused_kwargs):
    # Drops all the features
    return {}

def set_initial_node_states(node_set: tfgnn.NodeSet, node_set_name:str):
    """_summary_

    Args:
        node_set (tfgnn.NodeSet): _description_
        node_set_name (str): _description_
    """

    # Interaction
    if node_set_name=="Interaction":
        return  tf.keras.layers.Flatten()(node_set["summary"])
    # Scene
    if node_set_name=="Scene":
        return node_set["empty_state"]
    # Character
    if node_set_name=="Character":
        if train_flag[0]:
            return tf.keras.layers.Flatten()(node_set["name"])
        else:
           return node_set["empty_state"]
    # Relationship
    if node_set_name=="Relationship":
        if train_flag[1]:
            return tf.keras.layers.Flatten()(node_set["type"])
        else:
           return node_set["empty_state"]
    # Reason
    if node_set_name=="Reason":
        return node_set["empty_state"]
    # Emotion
    if node_set_name=="Emotion":
        if train_flag[2]:
            return tf.keras.layers.Flatten()(node_set["desc"])
        else:
           return node_set["empty_state"]
    # Attribute
    if node_set_name=="Attribute":
        if train_flag[3]:
            return tf.keras.layers.Flatten()(node_set["value"])
        else:
           return node_set["empty_state"]
    # Place
    if node_set_name=="Place":
        if train_flag[4]:
            return tf.keras.layers.Flatten()(node_set["desc"])
        else:
           return node_set["empty_state"]
    # Context
    if node_set_name=="Context":
        if train_flag[5]:
            return tf.keras.layers.Flatten()(node_set["desc"])
        else:
           return node_set["empty_state"]
    # Speech 
    if node_set_name=="Speech":
        if train_flag[6]:
            return tf.keras.layers.Flatten()(node_set["transcript"])
        else:
           return node_set["empty_state"]

def set_initial_edge_states(edge_set: tfgnn.EdgeSet, edge_set_name:str):
    """_summary_

    Args:
        edge_set (tfgnn.EdgeSet): _description_
        edge_set_name (str): _description_
    """

    # 5000 bins
    if edge_set_name=="involves" or edge_set_name=="linked_to":
        return tf.keras.layers.Embedding(5_000,32)(edge_set["hashed_role"])

def make_preprocessing_model(graph_tensor_spec, size_constraints):
    """Returns Keras model to preprocess a batched and parsed GraphTensor."""

    graph = input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec, name="model_input")

    # Convert input features to suitable representations for use on GPU/TPU.
    # Drop unused features (like id strings for tracking the source of examples).
    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=preprocess_node_features,
        edge_sets_fn=preprocess_edge_features,
        context_fn=drop_all_features)(graph)

    # Test
    assert "objectification" in graph.node_sets["Scene"].features

    ### IMPORTANT: All TF-GNN modeling code assumes a GraphTensor of shape []
    ### in which the graphs of the input batch have been merged to components of
    ### one contiguously indexed graph. There are no edges between components,
    ### so no information flows between them.
    graph = graph.merge_batch_to_components()

    # Optionally, pad to size_constraints (required for TPU).
    if size_constraints:
      graph, mask = tfgnn.keras.layers.PadToTotalSizes(size_constraints)(graph)
    else:
      mask = None

    # Split the label off the padded input graph.
    root_labels = tfgnn.keras.layers.Readout(
        node_set_name="Scene",
        feature_name="objectification"
    )(graph)

    # Remove the objectification tag from the Scene node
    graph = graph.remove_features(node_sets={"Scene": ["objectification"]})

    # Test
    assert "objectification" not in graph.node_sets["Scene"].features

    outputs = (graph, root_labels) if mask is None else (graph, root_labels, mask)

    return tf.keras.Model(input_graph, outputs)

def _get_dataset(split, *, shuffle=False, filter_fn=None,
                 input_context=None):
  
  filenames = []

  if split=="train":
    filenames = ["./tfrecords/mg_train_binary_full_completed.tfrecord"]
  elif split=="val":
    filenames=["./tfrecords/mg_val_binary_full_completed.tfrecord"]
  else:
    filenames=["./tfrecords/mg_val_binary_full_completed.tfrecord"]

  ds = tf.data.Dataset.from_tensor_slices(filenames)
  if input_context and input_context.num_input_pipelines > 1:
    ds = ds.shard(input_context.num_input_pipelines,
                  input_context.input_pipeline_id)
  if shuffle or split=="test":
    ds = ds.shuffle(len(filenames), seed=123)

  def interleave_fn(filename):
    ds = tf.data.TFRecordDataset(filename)
    if filter_fn is not None:
      ds = ds.filter(filter_fn)
    return ds
  ds = ds.interleave(
      interleave_fn, cycle_length=10,
      deterministic=False, num_parallel_calls=tf.data.AUTOTUNE)
  if shuffle:
    ds = ds.shuffle(10000, seed=123)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds

def _get_preprocessed_dataset(
    input_context, split_name, per_replica_batch_size, size_constraints):
  training = split_name == "train"

  ds = _get_dataset(split_name, shuffle=training,
                    input_context=input_context)
  if training:
    ds = ds.repeat()
  else:
     ds = ds.repeat()
  # There is no need to drop_remainder when batching, even for TPU:
  # padding the GraphTensor can handle variable numbers of components.
  ds = ds.batch(per_replica_batch_size)
  ds = ds.map(tfgnn.keras.layers.ParseExample(example_input_spec))

  if training and size_constraints:
    ds = ds.filter(functools.partial(tfgnn.satisfies_size_constraints,
                                     total_sizes=size_constraints))

  ds = ds.map(make_preprocessing_model(ds.element_spec, size_constraints),
              deterministic=False, num_parallel_calls=tf.data.AUTOTUNE)
  return ds

def _get_distributed_preprocessed_dataset(
    strategy, split_name, per_replica_batch_size, size_constraints):
  """Returns DistributedDataset and its per-replica element_spec."""
  return strategy.distribute_datasets_from_function(functools.partial(
      _get_preprocessed_dataset,
      split_name=split_name, per_replica_batch_size=per_replica_batch_size,
      size_constraints=size_constraints))


def build_model(graph_tensor_spec, message_dim=96, next_state_dim=544, dropout_rate=0.2, use_layer_normalization=False):
    """
    """

    # Create the input object
    graph = input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec, name="input")
    # Map features
    graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=set_initial_node_states, name="init_states")(graph)

    def dense(units, *, use_layer_normalization=False):
      # Set the regularizer
      #regularizer = tf.keras.regularizers.l2(l2_regularization)
      # Set the result
      result = tf.keras.Sequential([
          # Dense layer
          tf.keras.layers.Dense(
            units,
            activation="relu"),
          # Dropout layer
          tf.keras.layers.Dropout(dropout_rate, seed=123)])
      # Check if there's layer normalization
      if use_layer_normalization:
        # Add a layer normalization layer
        result.add(tf.keras.layers.LayerNormalization())

      return result

    def convolution(message_dim, receiver_tag):
      return tfgnn.keras.layers.SimpleConv(
          message_fn=dense(message_dim, use_layer_normalization=False),
          reduce_type="sum",
          receiver_tag=receiver_tag)


    def next_state(next_state_dim, use_layer_normalization, name):
      return tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim, 
                                                          use_layer_normalization=use_layer_normalization))
    

    # The graph udpate loop
    for _ in range(G_UPDATES):
      # Graph update
      graph = tfgnn.keras.layers.GraphUpdate(
          node_sets={
              "Interaction": tfgnn.keras.layers.NodeSetUpdate(
                  {"involves": convolution(message_dim, tfgnn.SOURCE), 
                   "has_speech": convolution(message_dim, tfgnn.SOURCE)
                   },
                  next_state(next_state_dim, use_layer_normalization, name="interaction")),
              "Character": tfgnn.keras.layers.NodeSetUpdate(
                  {"linked_to": convolution(message_dim, tfgnn.SOURCE), 
                   "expresses": convolution(message_dim, tfgnn.SOURCE), 
                   "is": convolution(message_dim, tfgnn.SOURCE)},
                  next_state(next_state_dim, use_layer_normalization, name="character")),
              "Scene": tfgnn.keras.layers.NodeSetUpdate(
                  {"location": convolution(message_dim, tfgnn.SOURCE), 
                   "circumstance": convolution(message_dim, tfgnn.SOURCE)}, 
                   next_state(next_state_dim, use_layer_normalization, name="scene")
              )
          })(graph)

    # Read out the hidden state of the root node
    root_states = tfgnn.keras.layers.Readout(node_set_name="Scene", 
                                             feature_name=tfgnn.HIDDEN_STATE)(graph)
    # Dense layer before the results 
    dense_layer = tf.keras.layers.Dense(256)(root_states)
    # Put a linear classifier on top
    logits = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(dense_layer)

    return tf.keras.Model(input_graph, logits)

def train_model(build_model_graph_tensor_spec, train_ds, valid_ds):
    """
    """

    with strategy.scope():
        # Build the model
        model = build_model(build_model_graph_tensor_spec)
        # Set the loss
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
        # Set the metrics
        metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(name="auc", from_logits=False)]
    
    num_training_samples = 417
    num_validation_samples = 261

    steps_per_epoch = num_training_samples // GLOBAL_BATCH_SIZE
    validation_steps = num_validation_samples // GLOBAL_BATCH_SIZE
    epochs = 200
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(1e-6, steps_per_epoch*epochs)
    #learning_rate = 1e-4

    # Compile the model
    # WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, 
    # please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        loss=loss, 
        weighted_metrics=metrics)
    
    class collect_garbage(tf.keras.callbacks.Callback): 
        def on_epoch_end(self, epoch, logs=None):
            # Collect the garbage at the end of the epoch
            collected = gc.collect()
            print(f"\nGarbage collector: collected {collected} objects.", end="\n\n")
    
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_binary_accuracy",
        mode="max",
        patience = PATIENCE_FACTOR, 
        verbose=1,
        restore_best_weights=True
    )

    # Model description
    model.summary()
    # Launch the training
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=valid_ds,
        validation_steps=validation_steps, 
        callbacks=[stop_early, collect_garbage()], 
        batch_size=GLOBAL_BATCH_SIZE
    )

    return history, model

def save_results(history):
    """_summary_
    """

    with open("./results/RoBERTA/results_GCN_binary.txt", "a+") as file:

        # Position
        position = len(history.history["loss"]) - PATIENCE_FACTOR - 1
        # Get the max values 
        max_val_acc = history.history["val_binary_accuracy"][position]
        max_val_loss = history.history["val_loss"][position]
        max_val_auc = history.history['val_auc'][position]
        max_train_acc = history.history["binary_accuracy"][position]
        max_train_loss = history.history["loss"][position]
        max_train_auc = history.history['auc'][position]

        # Set the list of info 
        info_list = ["Character", "Relationship", "Emotion", "Attribute", "Place", "Context", "Speech"]
        # First line to write 
        i = 0
        first_line = [{info_list[i]: train_flag[i]} for i in range(len(train_flag))]
        # Text to write 
        file.write(f"{first_line}\n")
        file.write(f"Training accuracy : {max_train_acc}\n")
        file.write(f"Validation accuracy: {max_val_acc}\n")
        file.write(f"Training AUC: {max_train_auc}\n")
        file.write(f"Validation AUC: {max_val_auc}\n")
        file.write(f"Training loss: {max_train_loss}\n")
        file.write(f"Validation loss: {max_val_loss}\n\n")


def generate_hidden_states_model(model):
    """_summary_
    """

    # Get the output of the readout layer
    layer_output_3 = model.layers[G_UPDATES+2].output
    # Generate the readout model
    readout_model = tf.keras.Model(inputs=model.input, outputs=layer_output_3)
    #readout_model.summary()
    
    return readout_model

def compute_score(sensitivity_matrix):
    """_summary_

    Args:
        sensitivity_matrix (_type_): _description_
    """

    # Compute the scale
    scale = 1.0 / (sensitivity_matrix.shape[0]+1)
    # Compute the scaled sensitivity 
    scaled_sensitivity = sensitivity_matrix * scale
    # Compute the global score 
    sensitivity_score = tf.reduce_mean(scaled_sensitivity).numpy()
    # Check if the sensitivity score is a nan
    if np.isnan(sensitivity_score):
            sensitivity_score=0.0

    return sensitivity_score

def run_inferences(readout_model, strategy):
    """_summary_
    """

    # Get the weight paths
    weights_paths = readout_model.get_weight_paths()

    # Get the entire validation set as a test set
    dataset = _get_distributed_preprocessed_dataset(
    strategy=strategy, split_name="test",
    per_replica_batch_size=261, size_constraints=None)

    for batch in dataset.take(1):
        with tf.GradientTape(persistent=True) as tape:
            output_at_readout = readout_model(batch[0])
            weights_at_l = readout_model.layers[-2].weights
            tape.watch(weights_at_l)
    
    # Initialize the dict of scores
    scores = dict()

    for i, weight in enumerate(weights_at_l):
        sensitivity_W = tape.gradient(output_at_readout, weight)
        #print(f"Sensitivity at : {list(weights_paths.keys())[i]} \n\n {sensitivity_W}.")
        if sensitivity_W is not None:
            # Get the sensitivity score
            score = compute_score(sensitivity_W)
            # Append to the dict of scores 
            scores[list(weights_paths.keys())[i]] = score
            #print(sensitivity_W.shape, end="\n\n")
            #print(score, end="\n\n")
    
    return scores


def save_scores(scores):
    """_summary_

    Args:
        scores (_type_): _description_
    """

    # Set the list of info 
    info_list = ["Character", "Relationship", "Emotion", "Attribute", "Place", "Context", "Speech"]
    # First line to write 
    i = 0
    first_line = [{info_list[i]: train_flag[i]} for i in range(len(train_flag))]

    with open("./results/ROBERTA/scores_GCN_binary.txt", "a+", encoding="utf-8") as file:
        # Text to write 
        file.write(f"{first_line}\n")
        # Loop through the scores
        for key,value in scores.items():
            # Write the values
            file.write(f"{key}: {value}\n")
        file.write("\n\n")
        # Display a message 
        print("Sensitivity scores saved.", end="\n\n")


def main():
    """_summary_

    Returns:
        _type_: _description_
    """

    # Load the dataset and the class names 
    dataset, class_names = load_dataset()
    # Setup the strategy
    strategy = setup_strategy()

    # Validation dataset
    validation_global_batch_size = GLOBAL_BATCH_SIZE
    assert validation_global_batch_size % strategy.num_replicas_in_sync == 0, "divisibility required"
    validation_replica_batch_size = validation_global_batch_size // strategy.num_replicas_in_sync
    print(f"Validation uses a global batch size of {validation_global_batch_size} "
      f"({validation_replica_batch_size} per replica).")
    validation_size_constraints = None

    # Training dataset 
    training_global_batch_size = GLOBAL_BATCH_SIZE
    assert training_global_batch_size % strategy.num_replicas_in_sync == 0, "divisibility required"
    training_replica_batch_size = training_global_batch_size // strategy.num_replicas_in_sync
    print(f"Training uses a batch size of {training_global_batch_size} "
        f"({training_replica_batch_size} per replica).")
    training_size_constraints = None

    # Get an example input spec
    global example_input_spec
    
    example_input_spec = tfgnn.create_graph_spec_from_schema_pb(GRAPH_SCHEMA)

    # Get the training dataset
    train_ds = _get_distributed_preprocessed_dataset(
    strategy, "train",
    training_replica_batch_size, training_size_constraints)
    # Get the validation dataset
    valid_ds = _get_distributed_preprocessed_dataset(
    strategy, "val",
    validation_replica_batch_size, validation_size_constraints)
 

    # Build the model
    build_model_graph_tensor_spec, *_ = _get_preprocessed_dataset(
    input_context=None, split_name="train",
    per_replica_batch_size=2, size_constraints=None).element_spec

    # Train the model 
    history, model = train_model(build_model_graph_tensor_spec, train_ds, valid_ds)
    # Write the results to a file
    save_results(history)

    # Interpretability sub-module 
    readout_model = generate_hidden_states_model(model)
    # Run inferences on the validation set
    scores = run_inferences(readout_model, strategy)
    # Save the results
    save_scores(scores)


if __name__ == "__main__":
   # Set the global parameter
   global train_flag
   # Loop through the training flags 
   for flag in TRAIN_FLAGS:
        # Assign the global training flag parameter
        train_flag = flag
        # Train the model
        try: 
            main()
        except Exception as error:
           # Print the error 
           print(f"Problem with this configuration: {train_flag}, \nError message : {error}", end="\n\n")
           traceback.print_exc()
    