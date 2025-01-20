#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from collections import defaultdict

def get_synynoms():
    """_summary_
    """

    # Set the list of synonyms 
    synonyms = defaultdict(list)

    # Add the words 
    synonyms["happy"] = ["happy", "cheerful", "delighted", "content", 
                         "enjoying", "overjoyed", "pleased", "satisfied", 
                         "thrilled", "playful", "playfull"]
    
    synonyms["worried"] = ["worried", "concerned", "distracted", "distressed", 
                           "tense", "anxious", "nervous", "worries", 
                           "nervious", "tense", "frantic", "overwrought", 
                           "strained"]
    
    synonyms["angry"] = ["angry", "annoyed", "bitter", "exasperated",
                         "furious", "impassioned", "irritated", "irritable", 
                         "mad", "outraged", "resentful", "sullen", 
                         "upset", "revolted", "irrritated", "snappy", 
                         "mad"]
    
    synonyms["surprised"] = ["surprised", "surprise", "astonished", 
                             "shocked", "shock", "dazed", "stunned", 
                             "startled"]
    
    synonyms["impressed"] = ["impressed", "amazed", "fascinated", "mesmerized"]
     
    synonyms["sad"] = ["sad", "heartbroken", "melancholic", "melancholy", 
                       "wistful", "depressed", "gloomy", "grieving", 
                       "grief", "suicidal", "regretful", "sullen", 
                       "apologetic"]
    
    synonyms["indifferent/uncaring"] = ["indifferent", "apathetic", "disinterested", "distant", 
                                        "uncaring", "unimpressed", "unenthusiastic", "uncompassionate", 
                                        "inconsiderate", "insensitive", "oblivious", "indiferent"]
    
    synonyms["kind"] = ["kind", "affectionate", "amiable", "compassionate", 
                        "considerate", "friendly", "sympathetic", "understanding", 
                        "frriendly", "generous", "hospitable", "freindly", 
                        "caring", "tender", "freindly", "welcoming"]
    
    synonyms["scared"] = ["scared", "afraid", "petrified", "panicky", 
                          "fearful", "terrified", "panics", "affraid"]
    
    synonyms["embarrassed"] = ["embarrassed", "ashamed", "humiliated", "insulted"]

    synonyms["sick"] = ["sick", "indisposed"]

    synonyms["menacing"] = ["menacing"]

    synonyms["joking"] = ["joking", "comical", "funny", "humorous", 
                          "ironic", "witty"]
    
    synonyms["rude"] = ["rude", "abusive", "impolite", "insulting", 
                        "petty", "racist", "homophobic", "immoral", 
                        "evil", "condescending", "sadistic", "offensive", 
                        "mean", "disrespectful", "hostile", "cheeky"]
    
    synonyms["arrogant"] = ["arrogant"]

    synonyms["bragging"] = ["bragging", "conceited", "boastful"]

    synonyms["confident"] = ["confident"]

    synonyms["scheming"] = ["scheming"]

    synonyms["crying"] = ["crying"]
    
    synonyms["hypocrite"] = ["hypocrite", "fake", "hypocritical", "dishonest", 
                             "perfidious", "insincere", "malicious", "traitorous"]
    
    synonyms["submissive"] = ["submissive", "servile", "obedient", "deferential", 
                              "subservient", "resigned"]

    synonyms["flirty/sexy/aroused"] = ["flirty", "sexy", "flirtatious", "horny", 
                                       "aroused", "seductive", "attracted", "frivolous", 
                                       "charming", "suave"]
    
    synonyms["clever"] = ["clever", "intelligent", "ingenious"]
    
    synonyms["drugged/drunk"] = ["drugged", "drunk"]

    synonyms["crazy"] = ["crazy", "delirious", "insane"]
    
    synonyms["calm"] = ["calm", "relaxed", "peaceful", "comfortable", 
                        "relaxes", "rested"]
    
    synonyms["uncomfortable"] = ["uncomfortable", "uneasy", "insecure"]

    synonyms["hesitant"] = ["hesitant", "unsure", "indecisive"]

    synonyms["obliging"] = ["obliging"]

    synonyms["noisy"] = ["noisy", "rowdy", "loud", "screaming"]
    
    synonyms["brave/bold"] = ["brave", "daring", "adventurous", "defiant"]
    
    synonyms["talkative"] = ["talkative", "chatty"]
    
    synonyms["thoughtful"] = ["thoughtful", "careful", "cautious", "attentive", 
                              "focused", "vigilant"]
    
    synonyms["appreciative"] = ["appreciative", "grateful"]
    
    synonyms["tired"] = ["tired", "exhausted", "sleepy"]

    synonyms["shy"] = ["shy", "timid", "reserved"]
    
    synonyms["motivated"] = ["motivated", "determined", "dedicated", "determind", 
                             "resolved", "eager"]
    
    synonyms["mocking"] = ["mocking"]

    synonyms["attracted"] = ["attracted", "allured"]
    
    synonyms["aggressive"] = ["aggressive", "aggresive"]

    synonyms["bossy"] = ["bossy", "dictatorial", "controlling"]

    synonyms["romantic"] = ["romantic", "loving"]

    synonyms["puzzled"] = ["puzzled"]
    

    return synonyms


def get_synonyms_attributes():
    """_summary_
    """

    # Set the list of synonyms 
    synonyms = defaultdict(list)

    synonyms["owner/CEO"] = ["owner", "record store owner", "store owner", "casino owner",
                             "CEO"]

    synonyms["manager"] = ["manager", "hotel manager", "casino manager", "building manager"]

    synonyms["doctor"] = ["doctor", "doctors", "child psychologist", "psychologist", 
                          "dentist"]

    synonyms["actor/actress"] = ["actor", "actress", "actor/actress"]

    synonyms["professor/instructor"] = ["professor", "professor of defense against the dark arts", "teacher", 
                                        "trainer", "tennis instructor", "yoga instructor", "sports coach"]

    synonyms["athlete"] = ["professional tennis player", "boxer", "hockey player"]

    synonyms["sick"] = ["sick", "ill"]

    synonyms["attractive"] = ["attractive", "handsome", "beautiful"]

    synonyms["assistant"] = ["assistant", "assistent"]

    synonyms["naked"] = ["naked", "half naked", "half-naked", "without pants", "nude"]

    synonyms["security officer/guard"] = ["security person", "security officer", "security guard", 
                                          "prison guard"]

    synonyms["police officer"] = ["police officer", "policeman", "police", "policemen"]

    synonyms["military officer"] = ["military officer", "military"]

    synonyms["adult"] = ["adult", "young adult"]

    synonyms["politician"] = ["politician", "senator", "government rep"]

    synonyms["reporter/journalist"] = ["reporter", "TV reporter", "journalist"]

    synonyms["artist"] = ["makeup artist", "artist", "comic book artist", "video artist", 
                          "make-up artist", "singer", "musician"]
    
    synonyms["businessman/woman"] = ["businessman/woman", "businessman"]

    synonyms["salesman/woman"] = ["saleswoman", "salesman/woman", "salesman"]

    synonyms["scantily dressed"] = ["scantily-dressed", "scantily dressed"]

    synonyms["fan"] = ["Philadelphia Eagles fan", "Cowboys America fan"]

    synonyms["mafia/drug lord"] = ["drug lord", "mafia boss"]

    synonyms["driver"] = ["driver", "cab driver", "bus driver"]

    synonyms["wizard"] = ["wizard", "magician"]

    synonyms["court workers"] = ["court clerk", "court bailiff"]

    synonyms["domestic worker"] = ["housekeeper", "nanny", "babysitter", "dog sitter"]

    synonyms["scientist"] = ["chemist", "nuclear physicist"]

    synonyms["waiter/waitress"] = ["waiter", "waitress", "waiter/waitress"]

    synonyms["sleeping"] = ["sleeping", "asleep"]

    synonyms["blond/blonde"] = ["blond", "blonde", "blond curly", "blond hair, blue eyes"]

    synonyms["fat"] = ["fat", "overweight"]

    synonyms["homosexual"] = ["lesbian", "gay"]

    return synonyms

def search_synonym(word: str):
    """_summary_
    """

    # Get the list of synonyms 
    synonyms = get_synynoms()

    # Loop through the synonyms
    for key, values in synonyms.items(): 
        # Check if the word belongs to the list of synonyms
        if word in values:
            return key
    
    return word

def search_synonym_attributes(word: str):
    """_summary_
    """

    # Get the list of synonyms 
    synonyms = get_synonyms_attributes()

    # Loop through the synonyms
    for key, values in synonyms.items(): 
        # Check if the word belongs to the list of synonyms
        if word in values:
            return key
    
    return word


# Get the list of synonyms 
#synonyms = get_synynoms()
#length = sum([len(values) for key,values in synonyms.items()])
#print(len(synonyms))
#print(length)

# Get the list of synonyms 
#synonyms = get_synonyms_attributes()
#length = sum([len(values) for key,values in synonyms.items()])
#print(len(synonyms))
#print(length)


