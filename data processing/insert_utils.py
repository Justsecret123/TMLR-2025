#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import re
import uuid


def insert_movies(movies_info:dict):
    """
    Generates the movies insertion script

    Parameters
    ----------
    movies_info : dict
        List of movies and their info (ID, title, year).

    Returns
    -------
    None.

    """

    # Create the file if it doesn't exit
    with open("insert_movies.osql", "w") as file:
        # Delete all the previously inserted movies
        file.write("TRUNCATE CLASS Movie UNSAFE\n")
        # Loop through the movies
        for id, movie_info in movies_info.items():
            # Create the query for the current movie and its info
            line = f'CREATE VERTEX Movie SET title="{movie_info["title"]}", year={movie_info["year"]}, mg_id="{id}"\n'
            # Append the query to the file
            file.write(line)


def insert_characters_and_features(clip_characters):
    """
    Generates the characters and their features insertion script

    Parameters
    ----------
    clip_characters : dict
        A dict containing characters for each clip.

    Returns
    -------
    None.

    """


    # Initialize the list of characters to insert
    # to avoid repetition
    inserted_characters = []
    # Initialize the list of features to insert
    # to avoid repetition
    inserted_features = []

    # Generate the characters insertion script 

    # Create the file if it doesn't exist 
    with open("insert_characters_and_features.osql","w") as file:
        # Delete all the previously inserted characters 
        file.write("TRUNCATE CLASS Character UNSAFE\n")
        # Delete all the previously inserted edges 
        file.write("TRUNCATE CLASS features UNSAFE\n")
        # Loop through the elements 
        for movie_id, clip in clip_characters.items():
            # Loop through the clip graphs 
            for clip_id, characters in clip.items():
                # Loop through the characters
                for character in characters:
                    if movie_id + "_" + character not in inserted_characters:
                        # Write the command to insert the current character
                        file.write(
                            f'CREATE VERTEX Character SET movie_id="{movie_id}", name="{character}"\n')
                        # Append to the list of inserted characters 
                        inserted_characters.append(movie_id + "_" + character)

                    if movie_id + "_" + character not in inserted_features:  
                        # Write the command to create the current edge
                        file.write(
                            f'CREATE EDGE features FROM (SELECT FROM Movie WHERE mg_id="{movie_id}") TO (SELECT FROM Character WHERE movie_id="{movie_id}" AND name="{character}")\n')
                        # Append to the list of inserted features
                        inserted_features.append(movie_id + "_" + character)


def insert_scenes_places_contexts(dataset):
    """
    Generates the scences (clips), places and contexts insertion script

    Parameters
    ----------
    dataset : Nx Graph
        The dataset containing all the clips graphs.

    Returns
    -------
    None.

    """

    # Inserted contexts
    inserted_contexts = [] 
    # Inserted places 
    inserted_places = []

    # Create the file if it doesn't exist 
    with open(f"insert_scenes_places_contexts.osql", "w") as file:
        # Delete all the scenes 
        file.write("TRUNCATE CLASS Scene UNSAFE\n")
        # Delete all the places
        file.write("TRUNCATE CLASS Place UNSAFE\n")
        # Delete all the contexts
        file.write("TRUNCATE CLASS Context UNSAFE\n")
        # Loop through the movies and clips 
        for movie, clips in dataset.items():
            # Loop through the scenes_id and graphs
            for clip_id, graph in clips.clip_graphs.items():
                # Get the original json
                clip_graph = graph.orig_graph_json
                # Get the description 
                scene_description = re.sub(r'["\n]', '',clip_graph["sentence_description"]).strip(
                ) if "sentence_description" in clip_graph else ""
                # Write the command to add this scene 
                file.write(
                    f'CREATE VERTEX Scene SET id="{movie+"_"+str(clip_id)}", description="{scene_description}", clip_id={clip_id}\n')
                # Write the command to add the edge between this scene and the movie 
                file.write(
                    f'CREATE EDGE has FROM (SELECT From Movie WHERE mg_id="{movie}") TO (SELECT FROM Scene WHERE id="{movie+"_"+str(clip_id)}" AND description="{scene_description}" AND clip_id={clip_id})\n')
                # Get the place
                place = re.sub(r'["\n]', '',clip_graph["scene"]).strip(
                ) if "scene" in clip_graph else None
                # Get the context 
                context = re.sub(r'["\n]', '',clip_graph["situation"]).strip(
                ) if "situation" in clip_graph else None

                # Check if the place is not null
                if place is not None and place != "":
                    # Check if the place has already been inserted
                    if place not in inserted_places:
                        # Write the command to add this place 
                        file.write(
                            f'CREATE VERTEX Place SET description="{place}"\n')
                        # Append the place to the list of inserted places 
                        inserted_places.append(place)
                    # Create the edge between this scene and its place 
                    file.write(
                        f'CREATE EDGE location FROM (SELECT FROM Scene WHERE id="{movie+"_"+str(clip_id)}" AND description="{scene_description}" AND clip_id={clip_id}) TO (SELECT FROM Place WHERE description="{place}")\n')
                # Check if the context has already been inserted 
                if context is not None and context != "":
                    # Check if the context has already been inserted
                    if context not in inserted_contexts: 
                        # Write the command to add this place 
                        file.write(
                            f'CREATE VERTEX Context SET description="{context}"\n')
                        # Append the context to the list of inserted contexts
                        inserted_contexts.append(context)
                    # Create the edge between this scene and its context
                    file.write(
                        f'CREATE EDGE circumstance FROM (SELECT FROM Scene WHERE id="{movie+"_"+str(clip_id)}" AND description="{scene_description}" AND clip_id={clip_id}) TO (SELECT FROM Context WHERE description="{context}")\n')


def insert_interactions(clip_interactions:dict):
    """
    Generates the clip interactions scripts

    Parameters
    ----------
    clip_interactions : dict
        A dict containing all the interactions data for each clip.

    Returns
    -------
    None.

    """

    # Create the inserted reasons list 
    inserted_reasons = []

    # Create the file if it doesn't exist 
    with open("insert_interactions.osql", "w") as file: 
        # Delete all the "involves" edges
        file.write("TRUNCATE CLASS involves UNSAFE\n")
        # Delete all interactions 
        file.write("TRUNCATE CLASS Interaction UNSAFE\n")
        # Delete all the reasons 
        file.write("TRUNCATE CLASS Reason UNSAFE\n")
        # Delete all the edges linking interactions and scenes 
        file.write(
            "DELETE FROM has where in IN (SELECT @rid from Interaction) AND out IN (SELECT @rid from Scene)\n")

        # Loop through the movie_ids and clips
        for movie_id, clips in clip_interactions.items():
            # Loop through the clip_ids and list
            for clip_id, interactions in clips.items(): 
                # Loop through the interactions for the current clip 
                for interaction in interactions:
                    # Set the interaction summary 
                    summary = re.sub(
                        r'["\n]', '', interaction["summary"]).strip()
                    # Write the command to insert the interaction
                    file.write(
                        f'CREATE VERTEX Interaction SET id="{movie_id+"_"+str(clip_id)}", summary="{summary}", start_time={interaction["start_time"]}, end_time={interaction["end_time"]}\n') 
                    # Check if there's a reason
                    if "reason" in interaction.keys():
                        # Check if the reason doesn't already exist in the database 
                        if (interaction["reason"],clip_id) not in inserted_reasons:
                            # Set the interaction reason 
                            reason = re.sub(
                                r'["\n]', '', interaction["reason"]).strip()
                            # Write the command to add this reason
                            file.write(
                                f'CREATE VERTEX Reason SET id="{movie_id+"_"+str(clip_id)}", description="{reason}"\n')
                            # Append the reason to the database 
                            inserted_reasons.append(
                                (interaction["reason"],clip_id))
                            # Write the command to insert the edge between the interaction and its reason 
                            file.write(
                                f'CREATE EDGE why FROM (SELECT FROM Interaction WHERE id="{movie_id+"_"+str(clip_id)}" AND summary="{summary}" AND start_time={interaction["start_time"]} AND end_time={interaction["end_time"]}) TO (SELECT FROM Reason WHERE description="{reason}" AND id="{movie_id+"_"+str(clip_id)}")\n')

                        # Create the edge between this scene and the interaction
                        file.write(
                            f'CREATE EDGE has FROM (SELECT FROM Scene where id="{movie_id+"_"+str(clip_id)}") TO (SELECT FROM Interaction WHERE id="{movie_id+"_"+str(clip_id)}" AND summary="{interaction["summary"]}" AND start_time={interaction["start_time"]} AND end_time={interaction["end_time"]})\n')

                        # Create edges for interaction roles 
                        if "performed_by" in interaction["characters"].keys() and interaction["characters"]["performed_by"] != set(): 
                            # Loop through the character who perfomed the interaction
                            for element in interaction["characters"]["performed_by"]:
                                file.write(
                                    f'CREATE EDGE involves FROM (SELECT FROM Interaction WHERE id="{movie_id+"_"+str(clip_id)}" AND summary="{summary}" AND start_time={interaction["start_time"]} AND end_time={interaction["end_time"]}) TO (SELECT FROM Character WHERE movie_id="{movie_id}" AND name="{element}") SET role="perfomed_by"\n')

                        if "towards" in interaction["characters"].keys() and interaction["characters"]["towards"] != set():
                            # Loop through the character towards whom the interaction was directed
                            for element in interaction["characters"]["towards"]:
                                file.write(
                                    f'CREATE EDGE involves FROM (SELECT FROM Interaction WHERE id="{movie_id+"_"+str(clip_id)}" AND summary="{summary}" AND start_time={interaction["start_time"]} AND end_time={interaction["end_time"]}) TO (SELECT FROM Character WHERE movie_id="{movie_id}" AND name="{element}") SET role="towards"\n')


def insert_characters_and_emotions(dataset_emotions):
    """
    Generates the characters emotions insertions script

    Parameters
    ----------
    dataset_emotions : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # Initialize the verification lists
    inserted_emotions = []
    inserted_expresses = []
    inserted_during = []

    # Create the file if it doesn't exist 
    with open("insert_characters_emotions.osql", "w") as file:
        # Delete all the previously inserted "Expresses" edges
        file.write("TRUNCATE CLASS expresses UNSAFE\n") 
        # Delete all the previoulsy inserted Emotions 
        file.write("TRUNCATE CLASS Emotion UNSAFE\n")
        # Loop through movies and clips 
        for movie_id, clips in dataset_emotions.items():
            # Loop through the clip graphs 
            for clip_id, clip in clips.items():
                # Loop through the characters and emotions
                for character, emotions in clip.items():
                    # Loop through the emotions
                    for emotion in emotions: 
                        # Check if the attribute is not in the list 
                        if (emotion,character,clip_id) not in inserted_emotions:
                            # Create the command to insert this attribute
                            line_emotion = f'CREATE VERTEX Emotion SET description="{emotion}", id="{movie_id+"_"+character+"_"+str(clip_id)}"\n'
                            # Append the command to the file
                            file.write(line_emotion)
                            # Update the inserted emotions list
                            inserted_emotions.append((emotion,character,clip_id))

                        # Check if we're not inserting the same "expresses" edge once again
                        # for the current character and clip
                        if (emotion,character,clip_id) not in inserted_expresses:
                            # Create the command to insert the edge 
                            # between the current attribute and the corresponding character
                            line_edge = f'CREATE EDGE expresses FROM (SELECT FROM Character WHERE movie_id="{movie_id}" AND name="{character}") TO (SELECT FROM Emotion WHERE description="{emotion}" AND id="{movie_id+"_"+character+"_"+str(clip_id)}")\n'
                            # Append the command to the file
                            file.write(line_edge)
                            # Update the inserted "expresses" edges list 
                            inserted_expresses.append((emotion,character,clip_id))
                        # Check if we're not inserting the same "during" edge once again
                        # for the current character and clip
                        if (emotion,character,clip_id) not in inserted_during: 
                            # Create the command to insert the edge 
                            # between the current emotion and the corresponding scene
                            line_edge = f'CREATE EDGE during FROM (SELECT FROM Emotion WHERE description="{emotion}" AND id="{movie_id+"_"+character+"_"+str(clip_id)}") TO (SELECT FROM Scene WHERE id="{movie_id+"_"+str(clip_id)}")\n'
                            # Append the command to the file
                            file.write(line_edge)
                            # Update the inserted "during" edges list 
                            inserted_during.append((emotion,character,clip_id))


def insert_characters_and_attributes(dataset_attributes:dict):
    """
    Generates the characters attributes insertion script.

    Parameters
    ----------
    dataset_attributes : dict
        A dict containing the character's attributes for each clip.

    Returns
    -------
    None.

    """

    # Initialize the verification lists
    inserted_attributes = []
    inserted_edges = []

    # Create the file if it doesn't exist 
    with open("insert_characters_attributes.osql", "w") as file:
        # Delete all the previously inserted "Possesses" edges
        file.write("TRUNCATE CLASS possesses UNSAFE\n") 
        # Delete all the previoulsy inserted attributes 
        file.write(f'DELETE VERTEX Attribute WHERE attribute_type="character"\n')
        # Loop through movies and clips 
        for movie_id, clips in dataset_attributes.items():
            # Loop through the clip graphs 
            for clip in clips.values():
                # Loop through the characters and attributes
                for character, attributes in clip.items():
                    # Loop through the attributes 
                    for key,value in attributes.items(): 
                        # Check if the attribute is not in the list 
                        if (key,value) not in inserted_attributes:
                            # Create the command to insert this attribute
                            line_attribute = f'CREATE VERTEX Attribute SET name="{key}", value="{value}", data_type="string", attribute_type="character"\n'
                            # Append the command to the file
                            file.write(line_attribute)
                            # Update the inserted attributes list
                            inserted_attributes.append((key,value))

                        # Check if we're not inserting the same set of attributes
                        # for the current character
                        if (key,value,character) not in inserted_edges:
                            # Create the command to insert the edge 
                            # between the current attribute and the corresponding character
                            line_edge = f'CREATE EDGE possesses FROM (SELECT FROM Character WHERE movie_id="{movie_id}" AND name="{character}") TO (SELECT FROM Attribute WHERE name="{key}" AND value="{value}")\n'
                            # Append the command to the file
                            file.write(line_edge)
                            # Update the inserted edges list 
                            inserted_edges.append((key,value,character))


def insert_relationships(dataset_relationships:dict):
    """
    Generates the relationships insertion script

    Parameters
    ----------
    dataset_relationships : dict
        The dataset containing the relationships between characters, 
        for each movie and clip(scene).

    Returns
    -------
    None.

    """
    
    # Create the file if it doesn't exist 
    with open("insert_relationships.osql","w") as file:
        # Delete all the previously inserted edges
        file.write(
            "DELETE EDGE WHERE out IN (SELECT @rid FROM Relationship)\n")
        file.write("TRUNCATE CLASS linked_to UNSAFE\n")
        # Delete all the previously inserted relationships 
        file.write("TRUNCATE CLASS Relationship UNSAFE\n")
        # Loop through the reltionshops 
        for movie_id, clips in dataset_relationships.items():
            # Loop through the clips dict
            for clip_id, relationships in clips.items():
                # Loop through the relationships 
                for relationship in relationships: 
                    # Set the relationship id 
                    id = "mg" + str(uuid.uuid4())[0:11]
                    # Write the command to insert the relationship
                    file.write(
                        f'CREATE VERTEX Relationship SET type="{relationship["type"]}", id="{id}"\n')
                    # Write the command to insert the edge between the subject and the relationship 
                    file.write(
                        f'CREATE EDGE linked_to FROM (SELECT FROM Character WHERE name="{relationship["subject"]}" AND movie_id="{movie_id}") TO (SELECT FROM Relationship WHERE type="{relationship["type"]}" AND id="{id}") SET role="{relationship["subject_role"]}"\n')
                    # Write the command to insert the edge between the object and the relationship 
                    file.write(
                        f'CREATE EDGE linked_to FROM (SELECT FROM Character WHERE name="{relationship["object"]}" AND movie_id="{movie_id}") TO (SELECT FROM Relationship WHERE type="{relationship["type"]}" AND id="{id}") SET role="{relationship["object_role"]}"\n')
                    # Write the command to insert the edge between the relationship and the scene 
                    file.write(
                        f'CREATE EDGE during FROM (SELECT FROM Relationship WHERE type="{relationship["type"]}" AND id="{id}") TO (SELECT FROM Scene WHERE id="{movie_id+"_"+str(clip_id)}")\n')
    
