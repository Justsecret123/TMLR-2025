#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 09:58:20 2023

@author: ibrahimserouis
"""

import tensorflow as tf 
import tensorflow_gnn as tfgnn
import generate_graphs_STELLA_FULL as generate_graphs
import pickle
from collections import defaultdict

# Initialize the dataset 
nx_dataset = defaultdict(dict)

# Generate the training set

# Read the binary pickle file
with open("./data/nx_dataset_objectification_train_posneg_completed.pkl","rb") as file: 
    # Load dataset and store into the variable
    nx_dataset = pickle.load(file)
    # Get the train len 
    train_len = len(nx_dataset)
    print(f"\n\nTraining samples: {train_len} ")
    # Initialize the counter
    i = 0
    # Generate the training set
    with tf.io.TFRecordWriter("./tfrecords/STELLA/mg_train_binary_full_completed.tfrecord") as writer: 
        # Loop through the graphs
        for id, graph in nx_dataset.items():
            # Get the node sets dict
            node_sets = generate_graphs.get_node_sets_dict(graph, id)
            # Get the edge sets dicts
            source_list, target_list, properties = generate_graphs.get_edge_sets_dicts(graph, node_sets, id)
            # Get the node sets data 
            node_sets_data = generate_graphs.get_node_sets_data(node_sets)
            # Get the edge sets data
            edge_sets_data = generate_graphs.get_edge_sets_data(source_list, target_list, properties)
            # Build a graph 
            graph = tfgnn.GraphTensor.from_pieces(
                node_sets=node_sets_data, 
                edge_sets=edge_sets_data
            )
            # Set the context 
            # Write the graph 
            output = tfgnn.write_example(graph)
            # Write to the output file 
            writer.write(output.SerializeToString())

            print(f"{i}", end="\r")
            i+=1

# Generate the validation dataset

# Read the binary pickle file
with open("./data/nx_dataset_objectification_val_posneg_completed.pkl","rb") as file: 
    # Load dataset and store into the variable
    nx_dataset = pickle.load(file)
    # Get the val len 
    val_len = len(nx_dataset)
    print(f"\n\nValidation samples: {val_len} ")
    # Initialize the counter
    i = 0
    # Generate the training set
    with tf.io.TFRecordWriter("./tfrecords/STELLA/mg_val_binary_full_completed.tfrecord") as writer: 
        # Loop through the graphs
        for id, graph in nx_dataset.items():
            # Get the node sets dict
            node_sets = generate_graphs.get_node_sets_dict(graph, id)
            # Get the edge sets dicts
            source_list, target_list, properties = generate_graphs.get_edge_sets_dicts(graph, node_sets, id)
            # Get the node sets data 
            node_sets_data = generate_graphs.get_node_sets_data(node_sets)
            # Get the edge sets data
            edge_sets_data = generate_graphs.get_edge_sets_data(source_list, target_list, properties)
            # Build a graph 
            graph = tfgnn.GraphTensor.from_pieces(
                node_sets=node_sets_data, 
                edge_sets=edge_sets_data
            )
            # Set the context 
            # Write the graph 
            output = tfgnn.write_example(graph)
            # Write to the output file 
            writer.write(output.SerializeToString())

            print(f"{i}", end="\r")
            i+=1