#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import datetime
import re
from collections import defaultdict
import get_variables
import webvtt
from transformers import BertTokenizer
# Set the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Set the list of files in the volume
volume_files = ["tt0120338", "tt0240772", "tt0388795", "tt1568346", 
                "tt0147800", "tt0286106", "tt1045658", "tt1570728"]
# Set the list of files in the local setup
local_files = ["tt0068646", "tt0106918", "tt0109831", "tt0110912", 
               "tt0114924", "tt0119822", "tt0167404", "tt0212338", 
               "tt0375679", "tt0455824", "tt0467406", "tt0822832", 
               "tt0970416", "tt1189340", "tt1454029", "tt1798709", 
               "tt1907668", "tt1010048", "tt0146882", "tt0790636", 
               "tt1142988", "tt138582", "tt0073486"]

def get_successors(graph:dict, node_id):
    """Extracts neighbors (successors) for a specific node

    Args:
        graph (dict): clip graph
        node_id (int/string): node identifier

    Returns:
        neighbors_list (list): the node's neighbor list
    """

    # Initialize the neighbors list
    neighbors_list = list()
    # Loop through the edges
    for edge in graph["edges"]:
        # Check if the source is our node 
        if edge["source"] == node_id:
            neighbors_list.append(edge["target"])
    
    return neighbors_list


def get_predecessors(graph:dict, node_id):
    """Extracts predecessors for a specific node

    Args:
        graph (dict): clip graph
        node_id (int/string): node identifier

    Returns:
        predecessors_list (list): the node's predecessors list
    """

    # Initialize the neighbors list
    predecessors_list = list()
    # Loop through the edges
    for edge in graph["edges"]:
        # Check if the source is our node 
        if edge["target"] == node_id:
            predecessors_list.append(edge["source"])
    
    return predecessors_list


def get_timestamps(graph:dict, node_id):
    """
    Extract interaction timestamps for a specific node

    Args:
        graph (dict): clip graph
        node_id (int/string): node identifier

    Returns:
       timestamps (dict) : interaction timestamps as {start_time, end_time}
    """

    # Get the neighbors list
    neighbors_list = get_successors(graph,node_id)
    
    # Initialize the timestamps
    # There are rare cases in the dataset where the start and/or end are not available,
    # hence the use of default values
    timestamps = {"start_time": -1, "end_time": -1}

    # Loop through the neighbors list 
    for node in graph["nodes"]:
        # Check if the node is a neighbor
        if node["id"] in neighbors_list:
            # Check if it's a timestamp
            if "from_type" in node.keys() and "t_start" in node.keys() and "t_end" in node.keys() and node["type"] == "time" and node["from_type"] == "interaction":
                # Assign the start and end
                timestamps["start_time"] = node["t_start"] 
                timestamps["end_time"] = node["t_end"] 

    return timestamps


def get_reason(graph:dict, node_id):
    """_summary_

    Args:
        graph (dict): _description_
        node_id (_type_): _description_
    """
    
    # Get the successors list 
    successors = get_successors(graph, node_id)

    # Initialize the reason 
    reason = None

    # Loop through the nodes
    for node in graph["nodes"]: 
        # Check if the node is a successor
        if node["id"] in successors:
            # Check if it's a reason
            if "type" in node.keys() and "from_type" in node.keys() and node["type"] == "reason" and node["from_type"] == "interaction":
                # Assign the reason to this node
                reason = node["name"]
                break
    
    return reason


def get_characters_and_roles(graph:dict, node_id):
    """
    Extract characters and their respective role for a specific interaction

    Args:
        graph (dict): clip graph
        node_id (int/string): node identifier

    Returns:
       characters (dict) : characters within the interaction as {performed_by, towards, spectator}
    """

    # Get the predecessors list
    predecessors_list = get_predecessors(graph, node_id)
    # Get the neighbors list 
    successors_list = get_successors(graph, node_id)

    # Initialize the character 
    characters = defaultdict(set)

    # Loop through the nodes 
    for node in graph["nodes"]: 
        # Set the character name : remove extra spaces and quotes
        character_name = node["name"].strip().replace('"', "")
        # Step 1 : extract who performed the action
        # Check if the node is a predecessor
        if node["id"] in predecessors_list: 
            # Check if it's a character 
            if "type" in node.keys() and node["type"] == "entity":
                # Assign the entity to the node 
                characters["performed_by"].add(character_name)
        # Step 2 : extract towards who the action was directed 
        # Check if the node is a successor
        if node["id"] in successors_list: 
            # Check if it's a character
            if "type" in node.keys() and node["type"] == "entity":
                # Assign the entity to the node
                characters["towards"].add(character_name)
        
    # Check if the action has been done by both characters (in this case, there's no "towards")
    if characters["performed_by"] == characters["towards"]: 
        characters["towards"] = set()
    
    return characters


def get_emotions(graph:dict, successors):
    """
    Extracts all the emotions for for a specific character 
    
    Parameters
    ----------
    graph : dict
        The dict containing the clip graphs.
    successors : list
        List of the character's node successors.

    Returns
    -------
    emotions : list
        List of emotions for the specific character.

    """

    # Initialize the emotions list
    emotions = []

    # Loop through the graph 
    for node in graph["nodes"]: 
        # Check if the node is a successor of the character node 
        if node["id"] in successors:
            # Check if the node is an emotion 
            if "type" in node.keys() and node["type"] == "attribute" and "emo:" in node["name"]:
                # Extract the emotion 
                emotion = re.search(r"emo:(\w+)", node["name"])
                # Check if the extraction is not null 
                if emotion is not None:
                    # Append to the emotions lsit
                    emotions.append(emotion.group(1))

    return emotions


def get_attributes(graph:dict, successors):
    """
    Returns the attributes within the successors of a specific [character] node.
    
    Attributes types : 
    - Gender
    - Age 
    - Ethnicity 
    - Profession    
    - Appearance
    - Sexual orientation

    Places and incorrect attributes are not taken into account

    Parameters
    ----------
    graph : dict
        A dict containing the clip graph.
    successors : list
        The list of the aforementioned node successors.

    Returns
    -------
    attributes : list
        The list containing the character's attributes.

    """

    # Initialize the attributes list
    attributes = dict()

    # Set the equivalence list
    equivalences = {"gen": "gender", "eth": "ethnicity", "age": "age", "app": 
                    "appearance", "pro": "profession", "ori": "sexual orientation"} 

    # Loop through the graph 
    for node in graph["nodes"]: 
        # Check if the node is a successor of the character node 
        if node["id"] in successors:
            # Check if the node is an attribute but not an emotion
            if "type" in node.keys() and node["type"] == "attribute" and not ("emo:" in node["name"]):
                # Extract the attribute type and value
                attribute_type = re.search(r"^(.*):", node["name"])
                attribute = re.search(r":(.*)", node["name"])
                # Check if the extraction is not null 
                if attribute_type is not None and attribute is not None:
                    # Get the attribute type text
                    attribute_type = attribute_type.group(1).strip()
                    attribute = attribute.group(1).strip()
                    # Check if an equivalence exists for this attribute
                    if attribute_type in equivalences:
                        # Update with the equivalence 
                        attributes[equivalences[attribute_type]] = attribute

    return attributes


def get_subject(graph:dict, predecessors):
    """
    Returns the name of subject in a relationship.
    Example : X (subject) is the brother of Y (object).
    X = subject, Y = object

    Parameters
    ----------
    graph : dict
        The dict containing the clip graph.
    predecessors : list
        The list of predecessors.

    Returns
    -------
    character_name : string
        The name of the subject.

    """

    # Initialize the character name
    character_name = ""

    # Loop through the nodes 
    for node in graph["nodes"]:
        # Check if the node is in the list of predecessors and is a character
        if "type" in node.keys() and node["id"] in predecessors and "entity" in node["type"]:
            # Set the character name : remove extra spaces and quotes
            character_name = node["name"].strip().replace('"', "")
            # End the loop 
            break
    
    return character_name


def get_object(graph:dict, successors, subject):
    """
    Returns the name of object in a relationship.
    Example : X (subject) is the brother of Y (object).
    X = subject, Y = object   

    Parameters
    ----------
    graph : dict
        The dict containing the clip graph.
    successors : list
        The list of successors.
    subject : string
        The name of the relationship's subject.

    Returns
    -------
    character_name : string
        The name of the object in the relationship.

    """

    # Initialize the character name
    character_name = ""

    # Loop through the nodes 
    for node in graph["nodes"]:
        # Check if the node is in the list of neighbors and is a character
        if "type" in node.keys() and node["id"] in successors and "entity" in node["type"]:
            # Set the character name : remove extra spaces and quotes
            character_name = node["name"].strip().replace('"', "")
            # Check if it's not a repetition of the subject
            if character_name != subject:
                # End the loop 
                break
    
    return character_name


def process_interaction(summary:str, equivalences:dict, not_found):
    
    # Intialize the object 
    synonym = "0"
    
    # Remove unwanted characters from the summary 
    summary = summary.replace('"','')
    
    # Loop through the equivalences 
    for label,synonyms in equivalences.items():
        # Check if the summary is in the list of synonyms
        if summary in synonyms: 
            # Assign the label to the variable 
            synonym = label
            # Break the loop
            break
    
    # Control
    if synonym=="0":
        not_found[summary]+=1
        return f"not_found:{summary}",not_found
        
    
    return synonym, not_found


def process_relationship(name, equivalences, pairs, relationship_types):
    """
    Returns relationship equivalences, roles and types, 
    relative to the relationship label (name).

    Parameters
    ----------
    name : string
        Initial lable of the relationship.

    Returns
    -------
    subject_role : string
        Role of the subject in the relationship.
    object_role : string
        Role of the object in the relationship.
    relationship_type : string
        The type of the relationship.

    """


    # Initialize the subject role
    subject_role = ""
    # Initialize the object role
    object_role = ""
    # Initialize the relationship type
    relationship_type = ""


    # Loop through the equivalence list 
    for relationship, synonyms in equivalences.items():
        # Check if the relationship name is in the list of synonyms 
        if name in synonyms:
            # Assign the synonym to the subject role
            subject_role = relationship
            # Break the loop
            break
    
    # Set the object role 
    # Loop through the relationship pairs
    for elementA, elementB in pairs: 
        # Check if the subject role is the first element of the pair
        if subject_role == elementA:
            # Assign the second element of the pair to B
            object_role = elementB
            break
        # Check if the subject role is the second element of the pair
        elif subject_role == elementB:
            object_role = elementA
            break
    
    # Set the relationship type
    # Loop through the relationship types and their tags
    for relationship, values in relationship_types.items():
        # Check if the relationship is in the list of tags 
        if subject_role in values: 
            # Assign the type to this relationship 
            relationship_type = relationship
            # End the loop
            break
    
    return subject_role,object_role,relationship_type


def extract_movies_info(dataset_movies, movies_data):
    """
    Extracts all the movies info

    Parameters
    ----------
    dataset_movies : list
        List of movies IDs.
    movie_names : list
        List of movies IDS and their corresponding info (title,year...).

    Returns
    -------
    movies_info : dict
    
    Movies and their info
    
    Format : 
        {movie_ID : {title, year}}

    """
    
    # Initialize the object
    movies_info = defaultdict(dict)
    
    # Loop through the lines
    for line in movies_data:
        # Loop through the movies names
        for movie in dataset_movies:
            # Check if the line corresponds to the movie 
            if movie in line:
                # Search the text after the movie
                text = re.search(rf"(?<={movie}\s).*", line).group()
                # Get the title 
                title = re.search(r"^.*?(?=\b\d{4}\b)", text).group()
                # Get the year 
                year = re.search(r"\b\d{4}\b",text).group()
                # Append the results to the movie equivalences list
                movies_info[movie] = {"title": title.strip(), "year": year}
    
    return movies_info


def extract_characters(dataset:dict):
    """
    Extracts all the characters for each clip graph

    Parameters
    ----------
    dataset : dict
        The dataset containing clip graphs.

    Returns
    -------
    clip_characters : dict
    
    Characters for each clip, for each movie
        Format : 
            {movie
             {clip:
              {characters list}
              }
             }.
    """
    
    # Initialize the object 
    clip_characters = defaultdict(dict)
    
    # Loop through the movies 
    for movie in dataset: 
        # Loop through the clips 
        for clip in dataset[movie].clip_graphs.items():
            # Intialize the object
            clip_characters[movie][clip[0]] = set()
            # Loop through the nodes
            for node in clip[1].orig_graph_json["nodes"]: 
                # Check if the node is a character (Entity type)
                if "type" in node.keys() and node["type"] == "entity":
                    # Set the character name : remove extra spaces and quotes
                    character_name = node["name"].strip().replace('"', "")
                    # Add the character to the set
                    clip_characters[movie][clip[0]].add(character_name)
    
    return clip_characters

def convert_interaction_time_to_datetime(time:str):
    """_summary_

    Args:
        time (str): _description_

    Returns:
        _type_: _description_
    """

    # Set the time format
    time_format = "%H:%M:%S.%f"

    # Cast to datetime
    time = datetime.datetime.strptime(time, time_format)

    return time.time()


def get_scene_boundaries(file_path): 
    """_summary_
    """

    if file_path!="none":
        try:
            # Open the file
            scene_boundaries = open(f"{file_path+'scenes.txt'}", "r", encoding="utf-8")
            #print(file_path)
            # Read the lines
            scene_boundaries = scene_boundaries.read().split("\n")
            #print(scene_boundaries)
            # Separate columns 
            scene_boundaries = [line.split() for line in scene_boundaries]
            # Cast the values to their respective types 
            scene_boundaries = [[int(line[0]), 
                                int(line[1]), 
                                convert_interaction_time_to_datetime(line[2]),
                                convert_interaction_time_to_datetime(line[3]), 
                                int(line[4])] for line in scene_boundaries]
        except IndexError: 
            print(f"Problem with : {file_path+'scenes.txt'}.")
    else: 
        scene_boundaries = None

    return scene_boundaries

def get_duration(start, end): 
    """_summary_

    Args:
        start (_type_): _description_
        end (_type_): _description_
    """

    start = datetime.datetime.combine(datetime.date.today(), start)
    end = datetime.datetime.combine(datetime.date.today(), end)

    return end-start


def extract_interactions(dataset):
    """
    Extracts interactions data : characters, roles, timestamps, reason

    Parameters
    ----------
    dataset : Nx graph
        The dataset containing all the clip graphs.

    Returns
    -------
    clip_interactions : dict
        Clips and their interactions, for each movie
        Format : {movie:
                  {clip:
                   {interaction: 
                    {characters: list,
                     start_time, 
                     end_time, 
                     reason (if available)
                     }}
                   }
            }

    """

    
    # Initialize the object 
    clip_interactions = defaultdict(dict)
    
    # Initialize the stats object
    stats = defaultdict(int)
    
    # Not found count 
    not_found = defaultdict(int)
    
    # Get the list of equivalences based on the method 
    # from Kukleva et al.
    equivalences = get_variables.get_kukleva_merged_interactions()
    
    
    # Loop through the movies 
    for movie in dataset: 
        if movie in local_files: 
            file_path = f"../../Clips_features_extraction.nosync/Clips.nosync/{movie}/"
        elif movie in volume_files: 
            file_path = f"/Volumes/maxone/Clips/{movie}/"
        else:
            file_path = "none"
        # Loop through the clips 
        for clip in dataset[movie].clip_graphs.items():
            # Initialize the object 
            clip_interactions[movie][clip[0]] = list()
            # Get the scene_boundaries 
            scene_boundaries = get_scene_boundaries(file_path)
            if scene_boundaries is not None:
                # Read the scene file at this interaction
                scene_boundary = scene_boundaries[clip[0]-1]
                # Get the frame count
                frame_count = scene_boundary[1]-scene_boundary[0]
                # Get the total duration 
                duration = get_duration(scene_boundary[2], scene_boundary[3])
            # Loop through the nodes
            for node in clip[1].orig_graph_json["nodes"]: 
                # Check if the node is an interaction
                if "type" in node.keys() and node["type"] in("interaction","action"):
                    # Get the interaction timestamps
                    timestamps = get_timestamps(
                        clip[1].orig_graph_json, node["id"])
                    # Get the characters and their roles in the interaction 
                    characters = get_characters_and_roles(
                        clip[1].orig_graph_json, node["id"])
                    
                    # Check if there's at least two characters within the interaction
                    if len(characters["towards"])+len(characters["performed_by"])<2:
                        with open("interactions_reject.txt", "a+") as file:
                            file.write(f"Invalid interaction. Characters : {characters}.\n")
                    else:    
                        # Check if there are at least two characters 
                        # Get the interaction's reason
                        reason = get_reason(clip[1].orig_graph_json, node["id"])

                        #if node["name"].strip() in("asks where","asks question"):
                        #    print(movie,clip)
                        # Process the interaction summary 
                        summary, not_found = process_interaction(node["name"].strip(), equivalences, not_found)
                        
                        # Chefk it it's a valid interaction
                        if summary!="invalid/irrelevant" and "not_found" not in summary:
                            # Update the stats object 
                            stats[summary]+=1
                            # Set the start and end frame 
                            start_frame = -1
                            end_frame = -1
                            files = []
                            # Check if the scene is used in the dataset 
                            if scene_boundaries is not None and scene_boundary[4]!=0:
                                # Get the start frame 
                                start_frame = round(scene_boundary[0] + ((timestamps["start_time"]/duration.total_seconds())*frame_count)) if duration.total_seconds()!=0 else -1
                                end_frame = round(scene_boundary[0] + ((timestamps["end_time"]/duration.total_seconds())*frame_count)) if duration.total_seconds()!=0 else -1
                                # Compute the interaction duration 
                                interaction_duration = timestamps["end_time"] - timestamps["start_time"] if timestamps["start_time"]!=-1 else -1

                            #if start_frame!=-1 and end_frame!=-1:
                                # Set the files to expect
                            #    files = [f"../../Clips_features_extraction.nosync/Clips.nosync/{movie}/frames/{movie}_scene_{i}.jpg" for i in range(scene_boundary[0], scene_boundary[1]+1)]

                                #print(f"Start frame : {start_frame}. End frame : {end_frame}. Duration computed : {interaction_duration} / Scene boundaries : [{scene_boundary[0]},{scene_boundary[1]}] ")
                            # Add the interaction and its info
                            if reason is not None:
                                (clip_interactions[movie][clip[0]]).append(
                                    {
                                        "characters": characters,
                                        "summary": summary, 
                                        "start_time": timestamps["start_time"], 
                                        "end_time": timestamps["end_time"], 
                                        "frame_start": start_frame,
                                        "frame_end": end_frame,
                                        "image_files": files,
                                        "reason": reason
                                    })
                            else:
                                (clip_interactions[movie][clip[0]]).append(
                                    {
                                        "characters": characters,
                                        "summary": summary, 
                                        "start_time": timestamps["start_time"], 
                                        "end_time": timestamps["end_time"], 
                                        "frame_start": start_frame,
                                        "frame_end": end_frame, 
                                        "image_files": files
                                    })
                         
                        
            # Check if it's an empty interaction node 
            if clip_interactions[movie][clip[0]] == []:
                # Delete the entry if it's empty
                clip_interactions[movie].pop(clip[0])
                
    return clip_interactions, not_found, stats


def extract_characters_and_emotions(graph:dict):
    """
    Extracts emotions and the related characters

    Parameters
    ----------
    graph : dict
        The dict containing the clip graphs.

    Returns
    -------
    characters_emotions : dict
        Characters and their emotions.
        Format :
            {movie
             {clip{
                 character: [emotions list]
                 }}
            }

    """

    # Characters and their emotions 
    characters_emotions = defaultdict(dict)

    # Loop through the nodes
    for node in graph["nodes"]: 
        # Check if the node is a character
        if "type" in node.keys() and "entity" in node["type"]:
            # Get the nodes successors
            successors = get_successors(graph, node["id"])
            # Set the character name : remove extra spaces and quotes
            character_name = node["name"].strip().replace('"', "")
            # Extract the emotions for this character 
            characters_emotions[character_name] = get_emotions(
                graph, successors)

    return characters_emotions


def extract_characters_and_attributes(graph:dict):
    """
    Extracts the attributes for all characters in the clip graph.

    Parameters
    ----------
    graph : dict
        A dict containing the clip graph.

    Returns
    -------
    characters_attributes : dict
        List of characters and their attributes.

    """

    # Characters and their attributes
    characters_attributes = defaultdict(dict)

    # Loop through the nodes
    for node in graph["nodes"]: 
        # Check if the node is a character
        if "type" in node.keys() and "entity" in node["type"]:
            # Get the nodes successors
            successors = get_successors(graph, node["id"])
            # Set the character name 
            character_name = node["name"].strip().replace('"', "")
            # Extract the attributes for this character 
            characters_attributes[character_name] = get_attributes(
                graph, successors)

    return characters_attributes


def extract_relationships(graph:dict):
    """
    Extracts the relationships and their info.

    Parameters
    ----------
    graph : dict
        The dict containing the clip graph.

    Returns
    -------
    relationships : list
        The list containing relationships for the current clip graph.
        Format (elements): 
            dict{
                type: relationship_type, 
                subject: subject_name, 
                subejct_role: role, 
                object: object_name, 
                object_role: role
                }

    """

    # Initialize the list of relationships
    relationships = []
    
    # 
    # Get the quivalences
    equivalences = get_variables.get_merged_relationships()
    # Get the pairs of relationships
    pairs = get_variables.get_relationships_pairs()
    # Get the types of relationships
    relationship_types = get_variables.get_relationships_types()

    # Loop through the graph 
    for node in graph["nodes"]:
        # Check if the node is a relationship 
        if "type" in node.keys() and node["type"] == "relationship":
            # Set the relationship name 
            name = node["name"].strip().lower()
            # Get the predecessors of the node 
            predecessors = get_predecessors(graph,node["id"])
            # Extract the relationship subject
            character_subject = get_subject(graph,predecessors)
            # Get the successors of the node
            successors = get_successors(graph,node["id"])
            # Extract the relationship object
            character_object = get_object(graph,successors,character_subject)
            # Process the relationship 
            subject_role, object_role, relationship_type = process_relationship(
                name, equivalences, pairs, relationship_types)
            # Check if it's a relevant relationship
            if subject_role != "irrelevant" and relationship_type != "":
                # Append to the relationships set
                relationships.append(
                    {"type": relationship_type, 
                     "subject": character_subject, 
                     "subject_role": subject_role, 
                     "object": character_object, 
                     "object_role": object_role})
    
    return relationships

def extract_kukleva_relationships(graph:dict):
    """
    Extracts the relationships and their info.

    Parameters
    ----------
    graph : dict
        The dict containing the clip graph.

    Returns
    -------
    relationships : list
        The list containing relationships for the current clip graph.
        Format (elements): 
            dict{
                type: relationship_type, 
                subject: subject_name, 
                subejct_role: role, 
                object: object_name, 
                object_role: role
                }

    """

    # Initialize the list of relationships
    relationships = []
    
    # 
    # Get the quivalences
    equivalences = get_variables.get_kukleva_merged_relationships()

    # Loop through the graph 
    for node in graph["nodes"]:
        # Check if the node is a relationship 
        if "type" in node.keys() and node["type"] == "relationship":
            # Set the relationship name 
            relationship_class = node["name"].strip().lower()
            # Get the predecessors of the node 
            predecessors = get_predecessors(graph,node["id"])
            # Extract the relationship subject
            character_subject = get_subject(graph,predecessors)
            # Get the successors of the node
            successors = get_successors(graph,node["id"])
            # Extract the relationship object
            character_object = get_object(graph,successors,character_subject)
            # Check if it's a relevant relationship
            if character_subject!="" and relationship_class not in ("", "relationship", "client", "other", "pastor", "first date", "other family", "lives with", "argue about relationship"):
                # Append to the relationships set
                relationships.append(
                    {"class": relationship_class, 
                     "subject": character_subject, 
                     "object": character_object})
    
    return relationships


def extract_subtitles(subtitle_paths:dict):
    """
    

    Parameters
    ----------
    subtitle_paths : dict
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Intialize the object 
    dataset_speech = defaultdict(dict)
    # Loop through the subtitles dict 
    for movie in subtitle_paths.keys():
        # Loop through the files
        for file in subtitle_paths[movie]:
            # Set the current file
            current_file = f"../../../MovieGraphs_Data/Subtitles/clip_srt/{movie}/{file}"
            # Extract the scene 
            current_scene = int(re.search(r"scene-(\d+)",current_file).group(1))
            # Set the captions list for the file
            captions_list = []
    
            # Exception handling 
            try:
                # Loop through the captions 
                for caption in webvtt.read(current_file):
                    # Extract relevant data : text, start, end then
                    # add to the captions list
                    total_text = caption.text.split("-")
                    # Extract the caption text 
                    # Remove special characters
                    text = list()
                    
                    # Loop through the captions
                    for x in total_text: 
                        # Compute the value 
                        value = tokenizer.tokenize(x)
                        if value!=[]:
                            # Add to the list
                            text.append(value)

                    # Remove special characters
                    captions_list.append({
                        "transcript": text, 
                        "start_time": caption.start, 
                        "end_time": caption.end
                        })
                # Update the speeches for the current movie
                dataset_speech[movie][current_scene] = captions_list
            # Raise an exception
            except:
                print(f"Some encodings are invalid. Check your caption file : {current_file} ")
    
    return dataset_speech
    
def extract_subtitles_V2(subtitle_paths:dict):
    """
    

    Parameters
    ----------
    subtitle_paths : dict
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Intialize the object 
    dataset_speech = defaultdict(dict)
    # Loop through the subtitles dict 
    for movie in subtitle_paths.keys():
        # Loop through the files
        for file in subtitle_paths[movie]:
            # Set the current file
            current_file = f"../../../MovieGraphs_Data/Subtitles/clip_srt/{movie}/{file}"
            # Extract the scene 
            current_scene = int(re.search(r"scene-(\d+)",current_file).group(1))
            # Set the captions list for the file
            captions_list = []
    
            # Exception handling 
            try:
                # Loop through the captions 
                for caption in webvtt.read(current_file):
                    # Extract relevant data : text, start, end then
                    # add to the captions list
                    total_text = caption.text.split("-")
                    # Extract the caption text 
                    # Remove special characters
                    text = list()
                    
                    # Loop through the captions
                    for x in total_text: 
                        # Compute the value 
                        value = tokenizer.tokenize(x)
                        if value!=[]:
                            # Add to the list
                            text.append(" ".join(value))

                    #if len(text)>1:
                    #    text = [" ".join(x) for x in text]

                    # Remove special characters
                    captions_list.append({
                        "transcript": text, 
                        "start_time": caption.start, 
                        "end_time": caption.end
                        })
                # Update the speeches for the current movie
                dataset_speech[movie][current_scene] = captions_list
            # Raise an exception
            except:
                print(f"Some encodings are invalid. Check your caption file : {current_file} ")
    
    return dataset_speech

    