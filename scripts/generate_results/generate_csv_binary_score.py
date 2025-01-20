#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 14:35:21 2023

@author: ibrahimserouis
"""

# Setup the globals 
global input_file
global output_file

import argparse
import pandas as pd

# Initialize the list of simplified terms for GraphSAGE
WEIGHTS_DICT_SAGE = {
    "graph_sage._node_set_updates.Character._edge_set_inputs.expresses._pooling_transform_fn.kernel":"character_expresses_kernel", 
    "graph_sage._node_set_updates.Character._edge_set_inputs.expresses._pooling_transform_fn.bias":"character_expresses_bias", 
    "graph_sage._node_set_updates.Character._edge_set_inputs.expresses._transform_neighbor_fn.kernel":"character_expresses_neighbors", 
    "graph_sage._node_set_updates.Character._edge_set_inputs.is._pooling_transform_fn.kernel":"character_is_kernel",
    "graph_sage._node_set_updates.Character._edge_set_inputs.is._pooling_transform_fn.bias":"character_is_bias",
    "graph_sage._node_set_updates.Character._edge_set_inputs.is._transform_neighbor_fn.kernel":"character_is_neighbors",
    "graph_sage._node_set_updates.Character._edge_set_inputs.linked_to._pooling_transform_fn.kernel":"character_linked_to_kernel",
    "graph_sage._node_set_updates.Character._edge_set_inputs.linked_to._pooling_transform_fn.bias":"character_linked_to_bias",
    "graph_sage._node_set_updates.Character._edge_set_inputs.linked_to._transform_neighbor_fn.kernel":"character_linked_to_neighbors",
    "graph_sage._node_set_updates.Interaction._edge_set_inputs.involves._pooling_transform_fn.kernel":"interaction_involves_kernel",
    "graph_sage._node_set_updates.Interaction._edge_set_inputs.involves._pooling_transform_fn.bias":"interaction_involves_bias_neighbors",
    "graph_sage._node_set_updates.Interaction._edge_set_inputs.involves._transform_neighbor_fn.kernel":"interaction_involves_neighbors",
    "graph_sage._node_set_updates.Interaction._edge_set_inputs.why._pooling_transform_fn.kernel":"interaction_why_kernel",
    "graph_sage._node_set_updates.Interaction._edge_set_inputs.why._pooling_transform_fn.bias":"interaction_why_bias",
    "graph_sage._node_set_updates.Interaction._edge_set_inputs.why._transform_neighbor_fn.kernel":"interaction_why_neighbors",
    "graph_sage._node_set_updates.Interaction._edge_set_inputs.has_speech._pooling_transform_fn.kernel":"interaction_speech_kernel",
    "graph_sage._node_set_updates.Interaction._edge_set_inputs.has_speech._pooling_transform_fn.bias":"interaction_speech_bias",
    "graph_sage._node_set_updates.Interaction._edge_set_inputs.has_speech._pooling_transform_neighbor_fn.kernel":"interaction_speech_neighbors",  
    "graph_sage._node_set_updates.Scene._edge_set_inputs.features._pooling_transform_fn.kernel":"scene_features_kernel", 
    "graph_sage._node_set_updates.Scene._edge_set_inputs.features._pooling_transform_fn.bias":"scene_features_bias", 
    "graph_sage._node_set_updates.Scene._edge_set_inputs.features._transform_neighbor_fn.kernel":"scene_features_neighbors", 
    "graph_sage._node_set_updates.Scene._edge_set_inputs.location._pooling_transform_fn.kernel":"scene_location_kernel", 
    "graph_sage._node_set_updates.Scene._edge_set_inputs.location._pooling_transform_fn.bias":"scene_location_bias", 
    "graph_sage._node_set_updates.Scene._edge_set_inputs.location._transform_neighbor_fn.kernel":"scene_location_kernel", 
    "graph_sage._node_set_updates.Scene._edge_set_inputs.circumstance._pooling_transform_fn.kernel":"scene_circumstance_kernel", 
    "graph_sage._node_set_updates.Scene._edge_set_inputs.circumstance._pooling_transform_fn.bias":"scene_circumstance_bias",
    "graph_sage._node_set_updates.Scene._edge_set_inputs.circumstance._transform_neighbor_fn.kernel":"scene_circumstance_neighbors",
    "graph_sage._node_set_updates.Scene._edge_set_inputs.has._pooling_transform_fn.kernel":"scene_has_kernel", 
    "graph_sage._node_set_updates.Scene._edge_set_inputs.has._pooling_transform_fn.bias":"scene_has_bias", 
    "graph_sage._node_set_updates.Scene._edge_set_inputs.has._pooling_transform_neighbor_fn.kernel":"scene_has_neighbor"
}
    

def convert_to_float(line:str):
    """_summary_

    Args:
        line (str): _description_

    Returns:
        _type_: _description_
    """

    return float(line.split(":")[1].strip())

def convert_to_float_sage(line:str):
    """_summary_

    Args:
        line (str): _description_
    """

    # Get the parameter name 
    parameter_name = WEIGHTS_DICT_SAGE[line.split(":")[0].strip()] \
          if line.split(":")[0].strip() in list(WEIGHTS_DICT_SAGE.keys()) \
          else line.split(":")[0].strip()
    
    parameter_name = str.capitalize(parameter_name)
    # Get the parameter value
    parameter_value = float(line.split(":")[1].strip())

    return parameter_name, parameter_value



def convert_to_bool(line:str):
    """_summary_

    Args:
        line (str): _description_

    Returns:
        _type_: _description_
    """
    
    return line.strip()=="True"

def convert_parameters(line:str):
    """_summary_

    Args:
        line (str): _description_
    """
    
    # Cast the parameters to a list
    line = line[1:len(line)-1].split(",")
    # Character
    character = convert_to_bool(line[0].replace("'", "").replace("}","").split(":")[1])
    # Relationship
    relationship = convert_to_bool(line[1].replace("'", "").replace("}","").split(":")[1])
    # Emotion
    emotion = convert_to_bool(line[2].replace("'", "").replace("}","").split(":")[1])
    # Attribute
    attribute = convert_to_bool(line[3].replace("'", "").replace("}","").split(":")[1])
    # Place
    place = convert_to_bool(line[4].replace("'", "").replace("}","").split(":")[1])
    # Context
    context = convert_to_bool(line[5].replace("'", "").replace("}","").split(":")[1])
    # Speech
    speech = convert_to_bool(line[6].replace("'", "").replace("}","").split(":")[1])

    return {"character": character, 
            "relationship": relationship, 
            "emotion": emotion, 
            "attribute": attribute, 
            "place": place, 
            "context": context, 
            "speech": speech}

def get_data_dict_GCN(results): 
    """_summary_
    """

    # Create the values lists
    character_expresses_kernel = []
    character_is_kernel = []
    character_is_bias = []
    scene_location_kernel = []
    scene_location_bias = []
    scene_circumstance_kernel = []
    characters = []
    relationships = []
    emotions = []
    attributes = []
    places = []
    contexts = []
    speeches = []

    # Loop through the results 
    for i in range(0,len(results),7):
        # Get the results set 
        results_set = results[i:i+7]
        # Get the parameters list
        parameters = convert_parameters(results_set[0])
        # Append to the parameters lists 
        characters.append(parameters["character"])
        relationships.append(parameters["relationship"])
        emotions.append(parameters["emotion"])
        attributes.append(parameters["attribute"])
        places.append(parameters["place"])
        contexts.append(parameters["context"])
        speeches.append(parameters["speech"])
        # Append to the lists of values
        character_expresses_kernel.append(convert_to_float(results_set[1]))
        character_is_kernel.append(convert_to_float(results_set[2]))
        character_is_bias.append(convert_to_float(results_set[3]))
        scene_location_kernel.append(convert_to_float(results_set[4]))
        scene_location_bias.append(convert_to_float(results_set[5]))
        scene_circumstance_kernel.append(convert_to_float(results_set[6]))

    # Generate the corresponding dict
    data = {"Character": characters, 
            "Relationship": relationships, 
            "Emotion": emotions, 
            "Attribute": attributes, 
            "Place": places, 
            "Context": contexts, 
            "Speech": speeches, 
            "Char_expresses_kernel":character_expresses_kernel,
            "Char_is_kernel":character_is_kernel, 
            "Char_is_bias":character_is_bias,
            "Scene_location_kernel":scene_location_kernel,
            "Scene_location_bias":scene_location_bias, 
            "Scene_circumstance_kernel":scene_circumstance_kernel
            }
    
    return data

def get_data_dict_sage(results):
    """_summary_

    Args:
        results (_type_): _description_
    """

    # Create the values lists
    param1 = []
    param2 = []
    param3 = []
    param4 = []
    param5 = []
    param6 = []
    param7 = []
    param8 = []
    param9 = []
    param10 = []
    param11 = []
    param12 = []
    param13 = []
    param14 = []
    characters = []
    relationships = []
    emotions = []
    attributes = []
    places = []
    contexts = []
    speeches = []

    # Loop through the results 
    for i in range(0,len(results),15):
        # Get the results set 
        results_set = results[i:i+15]
        # Get the parameters list
        parameters = convert_parameters(results_set[0])
        # Append to the parameters lists 
        characters.append(parameters["character"])
        relationships.append(parameters["relationship"])
        emotions.append(parameters["emotion"])
        attributes.append(parameters["attribute"])
        places.append(parameters["place"])
        contexts.append(parameters["context"])
        speeches.append(parameters["speech"])
        # Append to the lists of values
        p1_name, p1_value = convert_to_float_sage(results_set[1])
        p2_name, p2_value = convert_to_float_sage(results_set[2])
        p3_name, p3_value = convert_to_float_sage(results_set[3])
        p4_name, p4_value = convert_to_float_sage(results_set[4])
        p5_name, p5_value = convert_to_float_sage(results_set[5])
        p6_name, p6_value = convert_to_float_sage(results_set[6])
        p7_name, p7_value = convert_to_float_sage(results_set[7])
        p8_name, p8_value = convert_to_float_sage(results_set[8])
        p9_name, p9_value = convert_to_float_sage(results_set[9])
        p10_name, p10_value = convert_to_float_sage(results_set[10])
        p11_name, p11_value = convert_to_float_sage(results_set[11])
        p12_name, p12_value = convert_to_float_sage(results_set[12])
        p13_name, p13_value = convert_to_float_sage(results_set[13])
        p14_name, p14_value = convert_to_float_sage(results_set[14])
        param1.append(p1_value)
        param2.append(p2_value)
        param3.append(p3_value)
        param4.append(p4_value)
        param5.append(p5_value)
        param6.append(p6_value)
        param7.append(p7_value)
        param8.append(p8_value)
        param9.append(p9_value)
        param10.append(p10_value)
        param11.append(p11_value)
        param12.append(p12_value)
        param13.append(p13_value)
        param14.append(p14_value)

    # Generate the corresponding dict
    data = {"Character": characters, 
            "Relationship": relationships, 
            "Emotion": emotions, 
            "Attribute": attributes, 
            "Place": places, 
            "Context": contexts, 
            "Speech": speeches, 
            p1_name: param1,
            p2_name: param2, 
            p3_name: param3,
            p4_name: param4, 
            p5_name: param5, 
            p6_name: param6,
            p7_name: param7, 
            p8_name: param8,
            p9_name: param9, 
            p10_name: param10,
            p11_name: param11,
            p12_name: param12,
            p13_name: param13,
            p14_name: param14
            }
    # 
    
    return data, [p1_name,p2_name,p3_name,p4_name,
                  p5_name,p6_name,p7_name,p8_name,p9_name ]



def create_parser():
    """_summary_
    """

    # Create the parser 
    parser = argparse.ArgumentParser(description="Creates a csv from the results text file.")

    # Add the arguments 
    parser.add_argument("-input", help=".txt input file")
    parser.add_argument("-output", help=".csv output path")
    parser.add_argument("-model", help="model type (GCN, graphsage)")

    return parser


def main(results, output_file, model): 
    """_summary_

    Args:
        results (_type_): _description_
    """

    # Initialize the dataset
    dataset = []

    # Get the data dict 
    if str.lower(model)=="gcn":
        data = get_data_dict_GCN(results)
        # Get the csv data 
        dataset = pd.DataFrame(columns=["Character", "Relationship", "Emotion", "Attribute", 
                                    "Place", "Context", "Speech", "Char_expresses_kernel", 
                                    "Char_is_kernel","Char_is_bias", "Scene_location_kernel", 
                                    "Scene_location_bias", "Scene_circumstance_kernel"], data=data)
    
    elif str.lower(model)=="sage":
        data, params = get_data_dict_sage(results)
        # Set the list of columns 
        columns = ["Character", "Relationship", "Emotion", "Attribute", 
                   "Place", "Context", "Speech"] + params

        # Get the csv data 
        dataset = pd.DataFrame(columns=columns, data=data)
    
    # Save the dataset 
    dataset.to_csv(output_file, index=False)



if __name__=="__main__":

    # Create the parser 
    PARSER = create_parser()
    # Parse the command line arguments 
    ARGS = PARSER.parse_args()
    # Get the variables as a dict(key, value)
    VARIABLES = vars(ARGS)

    input_file = VARIABLES["input"]
    output_file = VARIABLES["output"]
    model = VARIABLES["model"]

    # Open the file and jump to the main
    with open(input_file, "r", encoding="utf-8") as file: 
        # Read the lines 
        results = file.read().split("\n")
        # Remove the empty lines
        results = [result for result in results if result!=""]
        # Go to the main
        main(results, output_file, model)
