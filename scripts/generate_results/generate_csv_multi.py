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


def convert_to_float(line:str):
    """_summary_

    Args:
        line (str): _description_

    Returns:
        _type_: _description_
    """

    return float(line.split(":")[1].strip())

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

def get_data_dict(results): 
    """_summary_
    """

    # Create the values lists
    training_acc = []
    val_acc = []
    val_auc_0 = []
    val_auc_1 = []
    val_auc_2 = []
    val_auc_3 = []
    training_loss = []
    val_loss = []
    characters = []
    relationships = []
    emotions = []
    attributes = []
    places = []
    contexts = []
    speeches = []

    # Loop through the results 
    for i in range(0,len(results),9):
        # Get the results set 
        results_set = results[i:i+9]
        # Get the parameters list
        parameters = convert_parameters(results_set[0])
        # Append to the lists
        training_acc.append(convert_to_float(results_set[1]))
        val_acc.append(convert_to_float(results_set[2]))
        training_loss.append(convert_to_float(results_set[3]))
        val_loss.append(convert_to_float(results_set[4]))
        val_auc_0.append(convert_to_float(results_set[5]))
        val_auc_1.append(convert_to_float(results_set[6]))
        val_auc_2.append(convert_to_float(results_set[7]))
        val_auc_3.append(convert_to_float(results_set[8]))
        # Append to the parameters lists 
        characters.append(parameters["character"])
        relationships.append(parameters["relationship"])
        emotions.append(parameters["emotion"])
        attributes.append(parameters["attribute"])
        places.append(parameters["place"])
        contexts.append(parameters["context"])
        speeches.append(parameters["speech"])

    # Generate the corresponding dict
    data = {"Character": characters, 
            "Relationship": relationships, 
            "Emotion": emotions, 
            "Attribute": attributes, 
            "Place": places, 
            "Context": contexts, 
            "Speech": speeches, 
            "Training acc": training_acc, 
            "Val acc": val_acc, 
            "Val auc 0": val_auc_0,
            "Val auc 1": val_auc_1,
            "Val auc 2": val_auc_2,
            "Val auc 3": val_auc_3,
            "Training loss": training_loss, 
            "Val loss": val_loss}
    
    
    return data


def create_parser():
    """_summary_
    """

    # Create the parser 
    parser = argparse.ArgumentParser(description="Creates a csv from the results text file.")

    # Add the arguments 
    parser.add_argument("-input", help=".txt input file")
    parser.add_argument("-output", help=".csv output path")

    return parser


def main(results): 
    """_summary_

    Args:
        results (_type_): _description_
    """

    # Get the data dict 
    data = get_data_dict(results)
    # Get the csv data 
    dataset = pd.DataFrame({ key:pd.Series(value) for key, value in data.items() })
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

    # Open the file and jump to the main
    with open(input_file, "r", encoding='utf-8') as file: 
        # Read the lines 
        results = file.read().split("\n")
        # Remove the empty lines
        results = [result for result in results if result!=""]
        # Go to the main
        main(results)
