#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from collections import defaultdict
from get_variables import get_node_edges_matches
import tensorflow as tf
import tensorflow_gnn as tfgnn


def search_node(node_name:str, node_sets:defaultdict(list)): 
    """
    

    Parameters
    ----------
    node_name : str
        DESCRIPTION.
    node_sets : defaultdict(list)
        DESCRIPTION.

    Returns
    -------
    node_type : TYPE
        DESCRIPTION.
    position : TYPE
        DESCRIPTION.

    """

    # Initialize the variables
    node_type = -1
    position = -1

    # Loop through the items
    for node_set, nodes in node_sets.items():
        # Loop through the nodes 
        for node in nodes: 
            # Check if it's the right node
            if node_name in node.values(): 
                # Set the node type 
                node_type = node_set
                # Set the position
                position = node["position"]
                # End loop
                return node_type, position
    
    return node_type, position


def get_node_sets_dict(graph):
    """

    Parameters
    ----------
    graph : TYPE
        DESCRIPTION.

    Returns
    -------
    node_sets : TYPE
        DESCRIPTION.

    """
    
    # Set the node list 
    node_list = list(graph.nodes)
    
    # Initialize the node sets 
    node_sets = defaultdict(list)
    
    # Initialize the count
    i = defaultdict(int)
    
    # Loop through the nodes
    for node_name in node_list:
        # Get the node object 
        node_data = graph.nodes[node_name]
    
        # Check if the node is a scene, place or context 
        if node_data["type"] in ["Scene","Place","Context"]:
            # Check if it's a scene 
            if node_data["type"]=="Scene": 
                # Add an id then insert 
                node_sets["Scene"].append({"id": node_name+"_"+str(i["Scene"]), "position": 0})
            else:
                # Append to the node sets 
                node_sets[node_data["type"]].append({"description": node_name, "position": 0})
        else: 
            # Check if it's a character 
            if node_data["type"]=="character":
                # Append a character to the node sets
                node_sets["Character"].append({"name": node_name, "position": i["Character"]})
                # Update the character position within the graph tensor
                i["Character"]+=1
            # Check if it's an emotion 
            elif node_data["type"]=="Emotion": 
                # Append an emotion to the node sets
                node_sets["Emotion"].append({"description": node_name.split(":")[0], "position": i["Emotion"]})
                # Update the emotion position 
                i["Emotion"]+=1
            # Check if it's an interaction 
            elif node_data["type"]=="Interaction":
                # Append an interaction to the node sets 
                node_sets["Interaction"].append({"summary": node_name, "position": i["Interaction"]})
                # Update the interaction position
                i["Interaction"]+=1
            # Check if it's an attribute 
            elif "attribute" in node_data["type"]:
                # Append an attribute to the node sets 
                node_sets["Attribute"].append({"name": node_data["name"], "value": node_name.split(":")[0], "position": i["Attribute"]})
                # Update the attribute position 
                i["Attribute"]+=1
    
    return node_sets


def get_node_sets_data(node_sets:dict):
    """
    

    Parameters
    ----------
    node_sets : dict
        DESCRIPTION.

    Returns
    -------
    node_sets_data : TYPE
        DESCRIPTION.

    """

    node_sets_data = dict()

    for node_type, values in node_sets.items():
        # Get the features 
        node_features = defaultdict(list)
        # Size 
        sizes = tf.constant([len(values)])
        
        for element in values:
            for key,value in element.items(): 
                if key!="position": 
                    node_features[key].append(value)

        node_features = {key:tf.ragged.constant(list(value),dtype=tf.string) for key,value in node_features.items()}

        node_sets_data[node_type] = tfgnn.NodeSet.from_fields(
            sizes=sizes,
            features=node_features
        )
    
    return node_sets_data

def get_edge_sets_data(source_list:dict, target_list:dict, properties:dict):
    """
    

    Parameters
    ----------
    source_list : dict
        DESCRIPTION.
    target_list : dict
        DESCRIPTION.
    properties : dict
        DESCRIPTION.

    Returns
    -------
    edge_sets_data : TYPE
        DESCRIPTION.

    """

    # Initialize the variables
    edge_sets_data = dict()
    # Counter 
    i = 0
    # Get the edge matches 
    matches, edge_matches = get_node_edges_matches()

    # Loop through the sources
    for edge_type, adjacency in source_list.items():
        # Set the size/dimension
        sizes = tf.constant([len(adjacency)])

        # Check if there are edge attribute
        if edge_type=="involves" or edge_type=="linked_to":
            # Assign the edge sets data with features
            edge_sets_data[edge_type] = tfgnn.EdgeSet.from_fields(
                sizes=sizes, 
                adjacency=tfgnn.Adjacency.from_indices(
                    source=(edge_matches[edge_type]["source"], tf.constant(adjacency)),
                    target=(edge_matches[edge_type]["target"], tf.constant(list(target_list.items())[i][1]))
                ), 
                features={
                    edge_type: tf.ragged.constant(properties[edge_type])
                }
            )
        else:
            # Assign the edge sets data without features
            edge_sets_data[edge_type] = tfgnn.EdgeSet.from_fields(
                sizes=sizes, 
                adjacency=tfgnn.Adjacency.from_indices(
                    source=(edge_matches[edge_type]["source"], tf.constant(adjacency)),
                    target=(edge_matches[edge_type]["target"], tf.constant(list(target_list.items())[i][1]))
                )
            )
        # Increase the counter 
        i+=1

    return edge_sets_data

