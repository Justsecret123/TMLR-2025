#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:35:21 2023

@author: ibrahimserouis
"""

from collections import defaultdict
import tensorflow as tf
import tensorflow_gnn as tfgnn
from transformers import AutoModel, AutoTokenizer
from synonyms import search_synonym, search_synonym_attributes
from sklearn.preprocessing import normalize


# Edge type matches from nodes 
matches = dict()
matches["Scene/Place"] = "location"
matches["Scene/Context"] = "circumstance"
matches["Scene/Character"] = "features"
matches["Scene/Interaction"] = "has"
matches["Character/Attribute"] = "is"
matches["Character/Emotion"] = "expresses"
matches["Interaction/Character"] = "involves"
matches["Interaction/Speech"] = "has_speech"
matches["Character/Relationship"] = "linked_to"
matches["Scene/Frames"] = "framed_by"


# Edges sources and targets
# Edge matches
edge_matches = defaultdict(dict)
edge_matches["location"] = {"source": "Scene", "target": "Place"}
edge_matches["circumstance"] = {"source": "Scene", "target": "Context"}
edge_matches["features"] = {"source": "Scene", "target": "Character"}
edge_matches["has"] = {"source": "Scene", "target": "Interaction"}
edge_matches["is"] = {"source": "Character", "target": "Attribute"}
edge_matches["expresses"] = {"source": "Character", "target": "Emotion"}
edge_matches["involves"] = {"source": "Interaction", "target": "Character"}
edge_matches["linked_to"] = {"source": "Character", "target": "Character"}
edge_matches["has_speech"] = {"source": "Interaction", "target": "Speech"}
edge_matches["representation"] = {"source": "Scene", "target": "Speech"}
edge_matches["framed_by"] = {"source": "Scene", "target": "Frames"}


# Embedding model
model = AutoModel.from_pretrained("infgrad/stella-base-en-v2")
# Set the tokenizer
tokenizer = AutoTokenizer.from_pretrained('infgrad/stella-base-en-v2')
# Set the kernel initializer
initializer = tf.keras.initializers.GlorotUniform(seed=123)



def get_node_sets_dict(graph, id="test"):
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
                node_sets["Scene"].append({"id": id, "position": 0, "objectification": int(node_data["objectification"])})
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
                node_sets["Attribute"].append({"value": node_name.split(":")[0], "position": i["Attribute"]})
                # Update the attribute position 
                i["Attribute"]+=1
            # Check if it's a relationship 
            elif node_data["type"]=="Relationship":
                # Append a relationship to the node sets 
                try:
                    node_sets["Relationship"].append({"type": node_name.split(":")[0], "position": i["Relationship"]})
                    # Update the relationship position 
                    i["Relationship"]+=1
                except: 
                    print(f"Problem with graph {id}. Relationship type: {node_name}")
            # Check if it's a speech 
            elif node_data["type"]=="Speech": 
                # Append a speech to the node sets 
                node_sets["Speech"].append({"name": node_name, "transcript": node_data["transcript"].replace(" ' ", "'").replace(" .", "."), "position": i["Speech"]})
                # Update the speech position 
                i["Speech"]+=1
            # Check if it's a set of frames
            elif node_data["type"] in ("Frame", "Frames"):
                # Check if the node representation exists
                if "representation" in node_data:
                    # Append the node to the node sets 
                    node_sets["Frames"].append({"name": node_name, "representation": node_data["representation"], "position": i["Frames"]})
                    # Update the frames position 
                    i["Frames"]+=1
    
    return node_sets



def search_node(key, node_type, node_sets):
    """_summary_

    Args:
        key (_type_): _description_
        node_type (_type_): _description_
        node_sets (_type_): _description_
    """

    # Set the attributes to search by node type
    att_to_search = {"Character": "name", 
                     "Emotion": "description", 
                     "Attribute": "value", 
                     "Relationship": "type", 
                     "Speech": "name", 
                     "Frames": "name"}
    
    # Set the position 
    position = -1
    
    # Loop through the nodes 
    for node in node_sets[node_type]: 
        # Check 
        if node[att_to_search[node_type]]==key: 
            return node["position"]
    
    return position


def search_relationship(initial, target, graph): 
    """_summary_

    Args:
        orig_graph (_type_): _description_
    """
    
    # Initialize the variables 
    position = -1

    # Loop through the items
    for node_name, attributes in list(graph.nodes(data=True)):
        # Check if it's the right node : compound name
        if len(node_name.split(":"))>1 and attributes["type"]=="Relationship" and node_name==target:
            # Set the position 
            position = int(node_name.split(":")[1])
            break
        # Check if it's the right node : simple name 
        elif len(node_name.split(":"))==1 and attributes["type"]=="Relationship" and node_name==target:
            # Set the position 
            position = initial
            break

    return position


def get_edge_sets_dicts(graph, node_sets:defaultdict(dict), id="test"):
    """
    

    Parameters
    ----------
    graph : TYPE
        DESCRIPTION.
    node_sets : defaultdict(dict)
        DESCRIPTION.
    id : TYPE, optional
        DESCRIPTION. The default is "test".

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    source_list : TYPE
        DESCRIPTION.
    target_list : TYPE
        DESCRIPTION.
    properties : TYPE
        DESCRIPTION.

    """

    # Get the edge list 
    edges = graph.edges(data=True)
    # Source list
    source_list = defaultdict(list)
    # Initialize the target list
    target_list = defaultdict(list)
    # Initialize the properties list 
    properties = defaultdict(list)


    # Loop through the edges (node pairs)
    for source,target, att in edges: 
        # Check if the source is the scene 
        if source=="Scene":
            # Set the variables 
            source_position = 0
            target_position = 0
            # Check if it the target is the interaction 
            if att["type"]=="has": 
                # Set the edge type 
                edge_type = matches["Scene/Interaction"]
            # Location
            elif att["type"]=="location": 
                # Set the edge type 
                edge_type = matches["Scene/Place"]
            # Circusmtance
            elif att["type"]=="circumstance": 
                # Set the edge type 
                edge_type = matches["Scene/Context"]
            # Frames 
            elif att["type"]=="representation":
                # Set the edge type
                edge_type = matches["Scene/Frames"]
                # Search for the set of frames
                target_position = search_node(target, "Frames", node_sets)
            # Features
            else:
                # Set the edge type 
                edge_type = matches["Scene/Character"]
                # Search for the character 
                target_position = search_node(target,"Character",node_sets)  
            # Add the source to the list
            source_list[edge_type].append(source_position)
            # Add the target to the list 
            target_list[edge_type].append(target_position)
        # The source is not the scene
        else:
            # Set the variables 
            source_position = -1
            target_position = -1
            # Check if it's an emotion being expressed 
            if att["type"]=="expresses": 
                # Set the edge type
                edge_type = matches["Character/Emotion"]
                # Search for the source 
                source_position = search_node(source,"Character", node_sets)
                # Search for the target 
                target_position = search_node(target.split(":")[0], "Emotion", node_sets)
            # Check if it's a character attribute 
            elif att["type"]=="possesses": 
                # Set the edge type 
                edge_type = matches["Character/Attribute"]
                # Search for the source 
                source_position = search_node(source, "Character", node_sets)
                # Search for the target 
                target_position = search_node(target.split(":")[0], "Attribute", node_sets)
            # Check if it's an involvement in an Interaction
            elif att["type"]=="involves": 
                # Set the edge type 
                edge_type = matches["Interaction/Character"]
                # Set the source position 
                source_position = 0
                # Search for the target position 
                target_position = search_node(target, "Character", node_sets)
                # Extract the role and append to the list 
                properties[edge_type].append(att["role"])
            #  Check if it's a relationship
            elif att["type"]=="linked_to": 
                # Set the edge type 
                edge_type = matches["Character/Relationship"]
                # Search for the source position 
                source_position = search_node(source, "Character", node_sets)
                # Search for the target position 
                target_position = search_node(target, "Relationship", node_sets)
                target_position = search_relationship(target_position, target, graph)
                # Extract the role and append to the list 
                properties[edge_type].append(att["role"])
            # Check if it's a speech 
            elif att["type"]=="has_subs": 
                #continue
                # Set the edge type 
                edge_type = matches["Interaction/Speech"]
                # Set the source position
                source_position = 0
                # Search for the target position 
                target_position = search_node(target, "Speech", node_sets)
            
            if source_position!=-1 and target_position!=-1:
                # Add the source to the list
                source_list[edge_type].append(source_position)
                # Add the target to the list 
                target_list[edge_type].append(target_position)
            else:
                print(source_position,target_position)
                raise ValueError(f"Could not find nodes - Source:{source}-Target:{target}")

    return source_list,target_list,properties

def get_node_sets_data(node_sets:dict, id="test"):
    """_summary_

    Args:
        node_sets (dict): _description_

    Returns:
        _type_: _description_
    """

    #print(f"Node sets: \n\n {node_sets}", end="\r")

    node_sets_data = dict()

    for node_type, values in node_sets.items():
        # Get the features 
        node_features = defaultdict(list)
        # Size 
        sizes = tf.constant([len(values)])
        
        for element in values:
            for key,value in element.items(): 
                if key!="position": 
                    #print(f"Key: {key}, Value: {value}")
                    # Set the node value
                    node_value = value
                    # Search for the emotion synonym
                    if node_type=="Emotion": 
                        # Assign the synonym
                        node_value = search_synonym(value)
                    elif node_type=="Attribute": 
                        # Assign the synonym
                        node_value = search_synonym_attributes(value)
                    node_features[key].append(node_value)

        # Setup the node features
        features = dict()

        # Loop through the node features 
        for key,value in node_features.items(): 
            # Check if it's the transcript
            #if key!="transcript" and key!="representation": 
            #    features[key] = tf.ragged.constant(list(value))
            #print(f"Key: {key}", end="\n\r")
            if key=="objectification":
                features[key] = tf.constant(value)
            elif key=="representation":
                    #print(f"Sizes: {sizes}")
                    #print(f"Len value: {len(value)}")
                    #print(f"Representation value: {value}", end="\n")
                    #print(value)
                    if len(value)==1 and value==-1:
                        features[key] = tf.constant(value)
                    else:
                        if len(value)<=1:
                            #print(f"Sizes: {sizes}")
                            #print(f"Len value: {len(value)}")
                            #print(f"Representation value: {value}", end="\n")
                            features[key] = value
                        else:
                            #print("Second case")
                            features[key] = tf.ragged.stack(value)
            else:
                # Set the batch data 
                batch_data = tokenizer(
                        padding="longest",
                        return_tensors="pt",
                        max_length=1024,
                        truncation=True,
                        text=value
                )
                # Get the attention mask 
                attention_mask = batch_data["attention_mask"]
                # Perform inference
                embedding = model(**batch_data)
                # Process the embedding
                last_hidden = embedding.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
                vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                vectors = vectors.detach().numpy()
                vectors = normalize(vectors, norm="l2", axis=1)
                # Check the size of the output
                if len(value)<=1:
                    # Update the feature
                    features[key] = tf.constant(vectors)
                else:
                    # Update the feature
                    features[key] = tf.ragged.stack(vectors)

        node_sets_data[node_type] = tfgnn.NodeSet.from_fields(
            sizes=sizes,
            features=features
        )

    return node_sets_data

def get_edge_sets_data(source_list:dict, target_list:dict,  properties:dict):
    """_summary_

    Args:
        source_list (dict): _description_
        properties (dict): _description_
    """

    # Initialize the variables
    edge_sets_data = dict()
    # Counter 
    i = 0

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