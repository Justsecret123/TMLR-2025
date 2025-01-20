#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from collections import defaultdict


def get_merged_relationships():
    """
    Sets and returns the  list of equivalences/synonyms of any relationship label

    Returns
    -------
    equivalences : dict
        The list of equivalences

    """

    # Initialize the object
    equivalences = defaultdict(list)

    # Set the equivalences 

    # Friendship
    equivalences["friend"] = ["friend","best friend","friends","close friend"]

    # Family
    equivalences["parent"] = ["parent","mother",
                              "step-mother","father","step-father","mother-in-law",
                              "father-in-law","parent-in-law"]
    
    equivalences["child"] = ["child","foster-son"]
    equivalences["grandparent"] = ["grandparent"]
    equivalences["grandchild"] = ["grandchild"]
    equivalences["brother/sister"] = ["brother","sister","brother in law",
                                      "brother in-law","brother-in-law","sister-in-law"]
    equivalences["other family"] = ["sibling","family","other family"]
    equivalences["aunt/uncle"] = ["aunt/uncle","aunt","uncle"]
    equivalences["cousin"] = ["cousin","distant cousin"]
    equivalences["godchild"] = ["godson","goddaughter"]
    equivalences["godparent"] = ["godfather","godmother"]
    equivalences["nephew/niece"] = ["nephew","niece"]

    # Work 
    equivalences["colleague"] = ["colleague","collaborator","replacement"]
    equivalences["business partner"] = ["business partner",
                                        "sponsor","client","customer","cusomer","customers"]

    # Acquaintance/Casual 
    equivalences["acquaintance"] = ["one night stand","acquaintance",
                                    "family friend","neighbor","ex-neighbor","pastor","first date"]
    equivalences["schoolmate"] = ["classmates","classmate","schoolmate"]
    equivalences["stranger"] = ["stranger"]
    equivalences["housemate"] = ["roommate","lives with"]


    # Mentorship 
    equivalences["student"] = ["student","students","apprentice"]

    # Romantic
    equivalences["lover"] = ["life partner","lover","engaged","fiance",
                             "fiancee","girlfriend","boyfriend","mistress","wife","husband","spouse"]
    
    equivalences["ex-lover"] = ["ex-boyfriend","ex-girlfriend",
                                "ex-lover","ex-fiance","ex-spouse","ex-husband","divorced","couple"]

    # Dependency/caretaking
    equivalences["doctor"] = ["doctor","vet","psychiatrist"]
    equivalences["superior"] = ["superior",
                                "superior(work)","superior (work)","supervisor","interviewer"]
    equivalences["caretaker"] = ["nanny","babysitter","nurse"]
    equivalences["teacher/instructor"] = ["teacher",
                                          "instructor","mentor","trainer"]
    equivalences["owner"] = ["owner","owns"]
    equivalences["boss"] = ["employer of","boss"]
    equivalences["employee"] = ["employed by","worker","merchandiser",
                                "host","lawyer","guardian","agent","aide","interviewee"]
    equivalences["landlord"] = ["landlord"]
    equivalences["tenant"] = ["tenant"]
    equivalences["patient"] = ["patient"]


    # Hostile
    equivalences["dislike"] = ["dislikes","dislike"]
    equivalences["captive"] = ["hostage","slave"]

    # Rivalry 
    equivalences["rival/enemy"] = ["rival","competitor","opponent","antognist",
                                   "antagonist","enemy","killer","robber"]

    # Community 
    equivalences["enthusiast"] = ["supporters","supporter","fan"]

    # Irrelevant 
    equivalences["irrelevant"] = ["would like to know","relationship","argue about relationship","heard about",
                                  "knows by reputation","other","public official","operative system","alleged lover",
                                  "potential employer of","in love with"]

    return equivalences

def get_kukleva_merged_relationships():
    """
    

    Returns
    -------
    equivalences : TYPE
        DESCRIPTION.

    """
    
    equivalences = defaultdict(list)
    
    # Friend
    equivalences["friend"] = ["close friend", "family friend", "best friend", "friend", "friends"]
    # Colleague 
    equivalences["colleague"] = ["business partner", "collaborator", "colleague"]
    # Lover 
    equivalences["lover"] = ["lover", "spouse", "engaged", "fiancee", "girlfriend", "mistress", "couple", "fiance", "alleged lover", "boyfriend", "life partner", "in love with"]
    # Parent
    equivalences["parent"] = ["parent", "aunt/uncle", "aunt", "uncle", "mother-in-law", "grandparent", "father-in-law", "godfather", "family", "step-mother", "parent-in-law"]
    # Boss
    equivalences["boss/owner"] = ["boss", "employer of", "mentor", "interviewer", "employer", "potential employer of", "owns" ]
    # Sibling 
    equivalences["sibling"] = ["sibling", "cousin", "sister/brother-in-law", "sister-in-law", "brother-in-law", "brother in law", "distant cousin", "brother in-law", "sister"]
    # kbr 
    equivalences["kbr"] = ["knows by reputation", "would like to know", "supporters", "heard about"]
    # Enemy 
    equivalences["enemy"] = ["enemy", "antagonist", "competitor", "killer", "dislike", "rival", "dislikes", "opponent"]
    # Customer 
    equivalences["customer"] = ["customer", "client", "patient", "cusomer", "customers", "students", "student", "tenant"]
    # Child 
    equivalences["child"] = ["child", "foster-son", "grandchild", "niece", "nephew", "niece/nephew", "godson", "goddaughter"]
    # Acquaintance
    equivalences["acquaintance"] = ["acquaintance", "neighbor", "roommate", "ex-neighbor", "supporter", "classmate", "fan", "classmates", "one night stand", "schoolmate"]
    # Worker 
    equivalences["worker"] = ["employed by", "apprentice", "slave", "interviewee", "guardian", "agent", "aide", "worker", "merchandiser" ]
    # Manager
    equivalences["manager"] = ["teacher", "owner", "superior", "landlord", "public official", "lawyer", "vet", "sponsor", "host", "psychiatrist", 
                               "supervisor", "trainer", "babysitter", "doctor", "nanny", "instructor", "nurse", "superior (work)", "superior(work)"]
    # Ex-lover
    equivalences["ex-lover"] = ["ex-boyfriend", "ex-girlfriend", "ex-fiance", "ex-spouse", "divorced", "ex-lover"]
    # Stranger 
    equivalences["stranger"] = ["stranger", "operative system", "robber", "replacement", "hostage"]
    
    return equivalences

def get_relationships_pairs():
    """
    Sets and returns the relationship pairs when available

    Returns
    -------
    pairs : set
        Relationship pairs

    """

    # Initialize the object 
    pairs = set()

    # Add the pairs 
    pairs.add(("friend","friend"))
    pairs.add(("parent","child"))
    pairs.add(("grandparent","grandchild"))
    pairs.add(("brother/sister","brother/sister"))
    pairs.add(("aunt/uncle","nephew/niece"))
    pairs.add(("cousin","cousin"))
    pairs.add(("other family","other family"))
    pairs.add(("godchild","godparent"))
    pairs.add(("colleague","colleague"))
    pairs.add(("business partner","business partner"))
    pairs.add(("acquaintance","acquaintance"))
    pairs.add(("schoolmate","schoolmate"))
    pairs.add(("customer","seller"))
    pairs.add(("stranger","stranger"))
    pairs.add(("housemate","housemate"))
    pairs.add(("student","teacher/instructor"))
    pairs.add(("lover","lover"))
    pairs.add(("ex-lover","ex-lover"))
    pairs.add(("doctor","patient"))
    pairs.add(("superior","subordinate"))
    pairs.add(("owner","owned"))
    pairs.add(("boss","employee"))
    pairs.add(("landlord","tenant"))
    pairs.add(("captive","captor"))
    pairs.add(("rival/enemy","rival/enemy"))
    pairs.add(("enthusiast","acclaimed"))


    return pairs 


def get_relationships_types():
    """
    Sets and returns the relationship types. 

    Returns
    -------
    types : dict
        Types of relationships and their tags.

    """

    # Initialize the object 
    types = dict()

    # Add the types of relationships and their tags 

    types["friendship"] = ["friend"]

    types["family"] = ["other family","cousin","aunt/uncle","nephew/niece",
                       "brother/sister","child","parent","godchild","godparent"]

    types["work"] = ["colleague","customer","business partner"]

    types["acquaitance/casual"] = ["acquaintance",
                                   "classmate","housemate","stranger"]

    types["mentorship"] = ["student","teacher/instructor"]

    types["romantic"] = ["lover","ex-lover"]

    types["dependency/caretaking"] = ["doctor","owner",
                                      "caretaker","boss","captive","employee","superior"]

    types["hostile"] = ["rival/enemy"]

    types["community"] = ["enthusiast"]

    return types



def get_kukleva_merged_interactions(): 
    """
    Returns the merged interactions with the method
    detailed in arXiv:2003.13158v1 [cs.CV] 29 Mar 2020

    Returns
    -------
    equivalences : list
        The list of equivalences.

    """
    
    
    # Initialize the equivalence list
    equivalences = dict()
    
    equivalences["leaves"] = ["leaves","leave together","gets out of the house","leave","leaves with","leaves nurse's station","leaves the office","leaves alone", "leaves with a woman"]
    
    equivalences["kisses"] = ["kisses", "tries to kiss","kiss","kisses on cheek","kisses on the cheek", "kisses goodbye","kiss on bench","hugs and kisses", "starts to kiss", "air kisses", 
                            "stops kissing", "leans in to kiss", "kisses the cat", "kisses her", "kisses a girl"]
    
    equivalences["walks (with)"] = ["walks", "walks with","walk with","accompanies","walks to","walks down the hallway", "walks in the street",
                             "walks away","walks away from","takes off","walk together","walk the hallway","walk down the hallway",
                             "walks behind","walks to the car","walks toward","walk past","walks arm in arm with","sit by pool",
                             "walks upstairs","walk around","walks towards","walks by","walks over to","walks in on", "walks around", 
                             "walks around the house", "walks barefoot in the park", "marches with"]
    
    equivalences["walks away from"] = ["walks away from", "takes off"]
    
    equivalences["follows"] = ["follows","follow","follows after"]
    
    equivalences["hugs"] = ["hugs", "puts arm around", "hug", "start to cuddle", "cuddles", "embraces", "embrace", "cuddle", "takes in arm", "cuddles with"]
    
    equivalences["sits (near/with)"] = ["sits","sits near","sits at table","sits together","sit down","sits with","sit together",
                                        "sit at the table","sit at table","sit in a car","sit in a restaurant","sits close to",
                                        "sit at a table","sit near","sit","sit on couch","sit in restaurant","sitsr near","sit in bar",
                                        "sits next to","sit on bench","sit in kitchen","sit at kitchen","sits down","'sit in a bar"]
    
    equivalences["passes by"] = ["passes by", "passes in the street"]
    
    equivalences["comes home"] = ["comes home", "tries to enter the room", "arrives"]  
    
    equivalences["meets"] = ["meets", "bumps into"]
    
    equivalences["hits/fights with"] = ["hits","slaps","attacks","punches","hurts","kicks","hits in head","hits in face","spite",
                            "hits on","hits with elbow","hits in arm","hits under table", "strangles","attack","bites",
                            "kicks with foot","hits with fist","punches back","punches in belly","punches to the floor", 
                            "fights","wrestles with","fight in ring","fights with","fight","wrestle", "kicks him", "tries to punch", 
                            "fighting in the ring", "breaks bottle on a guy"]
    
    
    equivalences["approaches"] = ["approaches", "draws near to", "approach", "approach the broom", "approaches his bed", 
                                  "approach the vault", "gets closer to", "approaches girfriend", "approaches the passenger"]    
    
    equivalences["stays (with)"] = ["stays", "stands near", "stand near", "stays near", "stays with", "wait for",
                                    "stands on rooftop", "waits for", "spends time with", "waits for the medication"]
    
    equivalences["plays (with)"] = ["plays","plays with","plays cards","plays poker","play tennis","start to play","play poker",
                                    "plays basketball with","play a game","plays ping pong","play cards", "plays with bubbles", 
                                    "plays with her dolls", "plays with his sword", "plays a video game", "plays with toy car on belly", 
                                    "plays with puppies", "play with a ball", "plays with friend's baby"] 
    
    equivalences["wakes up"] = ["wakes up", "wake up"]
    
    equivalences["runs (with/towards)"] = ["runs", "runs to", "chase", "start to run", "runs with", "rushes", "chases", "runs after", "run", "run together", "run with", 
                            "runs through the hallway", "runs with team", "runs to the bench", "runs after her", "runs to the house", "runs upstairs", 
                            "runs through the church"]  
    
    equivalences["runs from"] = ["runs from", "runs away", "run away", "runs away from", "escapes from", "runs aways from", "runs away with a letter", "runs home"]
    
    equivalences["rides/drives (to/with)"] = ["rides", "rides with","drives", "picks up", "rides together", "moves van", "drive by subway", "drive in taxi", "drive in car", "rides the bicycle",
                             "drive on boat", "ride in car", "ride together", "ride horses", "drives car", "drives a car", "drives fast", "drives very fast", "drives too fast", 
                             "gives a ride to", "drives away", "rides a horse", "parks the car", "drives off", "rides horse", "drives with", "rides car with", "ride in a car", 
                             "rides the horse", "rides a bull", "drives in taxi", "drives his car", "drives the car", "drives a bike", "drives monster truck", "drives the tank", 
                             "driving together", "drives to", "gives a ride"]
    
    equivalences["holds"] = ["holds", "holds in arms", "hold hands", "holds his arm",
                             "holds", "catches", "holds hand", "holds hands"]
    
    equivalences["visits (someone/somewhere)"] = ["visits","visit an office","visit","comes to see"]
    
    equivalences["dances (with)"] = ["dances with","dance", "dance with", "dances", "starts to dance", "dances for", "tap-dances", "dancing"]
    
    equivalences["pushes"] = ["pushes", "touches","pushes away", "pushes to", "go to the room "] 
    
    equivalences["shoots/points gun (at)"] = ["shoots", "shoot at", "shoots at", "points gun", "points a gun at", "points gun at", "shoots at door", "fires a gun", "fires gun", "shoots at enemy"]
    
    equivalences["eats (with)"] = ["eats with","drinks with", "feeds", "has dinner","have dinner", "have lunch","toasts with", "eats", "eats breakfast", "eats dinner", "eats snacks", 
                                   "eats with his hands", "eats with her hands", "eats lunch", "dines with", "eat together"]
    
    equivalences["takes"] = ["takes", "grabs", "buys from", "takes from", "takes away"]
    
    equivalences["opens"] = ["opens","opens the door for","opens door","opens door for"]
    
    equivalences["brings (for)"] = ["brings", "carries", "serves", "brings food for", "brings coffee to", "brings tea to"]   
    
    equivalences["points at/to"] = ["points at", "point at", "points", "points to"]		
			
    equivalences["hides from"] = ["hides from"]		
				
    equivalences["tries"] = ["tries"]	
    
    equivalences["puts"] = ["puts"]			
    
    equivalences["throws (at/to)"] = ["throws", "throws at", "throws scarf at", "throws key to", "throws club at", 
                                      "throws flowers to", "throws rope at", "throws car keys", "throws shirt at", 
                                      "throws pie at"]    
    
    equivalences["pulls"] = ["pulls", "drags", "drag", "pulls her in", "pulls him"]   
    
    equivalences["sends away"] = ["sends away"]		
				
    equivalences["has sex (with)"] = ["has sex", "has sex with", "sex", "makes love to", "has an orgasm"]		
			
    equivalences["fishes"] = ["fishes", "fishing"]						
    
    equivalences["gives (to)"] = ["gives", "gives to", "pays", "gives a gift","gives gift to",
                                  "gives phone to","gives pen to","gives cigarettes to","gives the ball","gives cookies to",
                                  "gives money to", "gives phone", "sells to", "gives a drink to","gives coffee",
                                  "gives a coat to","give to","give somthing", "gives a packet to","gives medical forms to",
                                  "gives the key","gives the lamp","gives the wand","gives another wand","gives flower to",
                                  "gives a bat","gives card to","gives keys to","gives briefcase to","gives coffee to",
                                  "gives the ball back","gives him gums","gives the bag","gives a bag","gives letters to",
                                  "gives letter","gives drawing to","gives device","gives drink to","gives coat to","gives file to", 
                                  "gives money", "gives papers", "gives magazine to", "gives rings to", "gives a drink to", 
                                  "gives CD", "gives letter to", "gives her card"
                                  ]
    
    equivalences["explains (to)"] = ["explains", "explains to", "instructs","explains about situation", "explains about current situation",
                                     "explains situation", "excuses self", "teaches", "describes", "expalins to", "describes to", 
                                     "explains problem to", "explains joke", "explains why is he stuck in his job", "explains a plan to",
                                     "explains about behavior", "explains emotion", "explains concern", "explains behavior", "explains action to", 
                                     "explains problem", "explains self", "explains about past experience", "explains past experience", "explains plan", 
                                     "explains future", "explains why she played the recording", "explains what he learned from break-up", 
                                     "explains how's he good at dates", "explains how he pretends to be sensitive", "explains how to make great tape", 
                                     "gives instructions"]  
    
    equivalences["answers (to)"] = ["answers","responds","replies","responds to","replies to","replies about rape", "answers about situation"
                                    "answer","answers about plan","answers about self","asnwers", "answers about work", "answers about someone", 
                                    "answers about friend", "answers question", "shyly answers", "answer", "answers about situation"]
    
    equivalences["orders"] = ["orders", "requests","demands", "begs", "orders", "about work", "orders about plan", "ordersabout work", 
                              "gives orders about plan", "orders about emergency", "orders to fire", "orders about work", "gives orders", 
                              "gives orders to", "orders servants to go move", "orders van driver to get out", "orders his team to get out", 
                              "orders by microphone"]   
    
    equivalences["greets"] = ["greets","welcomes","shakes hands","waves at","says goodbye", "shake hands", "shake hands with",
                              "waves to","waves head to","greet","waves goodbye","says goodbye to","shakes hand","salutes",
                              "says goodnight to","says bye to", "says goodnight", "shakes hand with", "shakes hands with", "greets goodbye", 
                              "greets guests", "greets people in the room", "welcomes back", "says goodbye to friends"]   
    
    equivalences["reassures"] = ["reassures", "supports", "comforts", "consoles", "tries to reassure", "tries to comfort", "reassiures"]  
    
    equivalences["suggests/offers (to/something)/gives opinion"] = ["suggests", "suggests to", "advises", "warns", "invites", "offers", "offers to help",
                                "offers food to","gives opinion", "proposes", "corrects", "offers to", "asks out", "offers flowers to",
                                "calls out to", "proposes to", "offers drink to", "offers alcohol to", "offers money", "offers help to", 
                                "offers sandwiches to", "offers coffee to", "offer", "offers to pay", "offers a drink", "offers a ride",
                                "offers to buy a drink", "offers ice cream", "offers to pick him up", "offers water to", "offers a seat to", 
                                "recommends", "offers the bet", "offers money to", "offers deal to", "offers tea to", "suggests\xa0 to", 
                                "gives suggestions", "suggest to", "suggests solution", "offers his business card", "expresses opinion", 
                                "gives opinion to", "tells opinion", "corrects her", "gives opinion about relationships", "gives opinion about", 
                                "gives opinion about their relationship", "gives opinion about Colin", "gives opinion about something"]
    
    equivalences["compliments/seduces"] = ["compliments", "praises", "reciprocates compliment", "compliments hands",
                                           "flatters", "admires", "sympathizes with", "flirts with","flirting",
                                           "flirts", "flirt", "thanks", "tries to seduce", "seduced him", "strips for", 
                                           "sympathises with"]    
    
    equivalences["helps"] = ["helps", "assists","tries to help", "sets free", "frees", "uncuffs", "helps boyfriend to get dressed", "helps kids up the ladder"] 
    
    equivalences["encourages"] = ["encourages", "encourages others", "motivates"]			
				
    equivalences["apologizes"] = ["apologizes to", "apologize","apologizes","apologizing"]	
					
    equivalences["reminds"] = ["reminds"]						
    
    equivalences["convinces"] = ["convinces", "persuades", "urges", "insists", "tries to convince", "calls for", "tries to persuade", "nudges"] 
    
    equivalences["assures (to)"] = ["assures", "promises", "swears", "swears to"]
    
    equivalences["congratulates/celebrates/claps"] = ["congratulates","applauds", "applaud", "applaud for","applauds for","applaud to",
                                     "cheers","cheers for","celebrates","applauds to","toasts to","claps for", "applaudes", "claps",
                                     "cheer for", "cheer on", "cheer","cheer at","awards", "wishes luck", "claps hands", "claps and cheers",
                                     "cheers and claps", "celebrates with", "celebrate", "tries to cheer up", "clap hands", "clap", 
                                     "clap at", "clap to"]  
    
    equivalences["leads/coaches"] = ["leads", "guides", "directs", "coaches"]
    
    equivalences["agrees (with/to)"] = ["agrees","confirms to","agrees to","agrees with comment","confirms","accepts","allows", 
                                        "gives permission","agrees","acceptsrequest","agree","approves","agrees with","agree with", 
                                        "agrees to gather intel", "agrees to murder", "agrees to work with", "accepts offer", 
                                        "accepts the bet", "accepts help from", "accepts invitation", "reluctantly agrees with"] 
    
    equivalences["admits"] = ["admits", "admits to", "confesses to", "admits he wanted to do nothing all day", "admits she invited his fiancee", "admits he drank", "admits she is", 
                              "admits he dreams of killing the boss", "admits he screwed up again", "admits they are not arrested", "confesses"]
    
    equivalences["announces (to)"] = ["announces", "breaks news to", "declares love to", "announces to", "announces shifts for singing",
                                       "declares love", "states love for", "states love to"] 
    
    equivalences["receives"] = ["news from", "receives news from", "receives news about", "receives message from"]   
    
    equivalences["wishes"] = ["wishes", "hopes for"]
    
    equivalences["saves"] = ["saves", "defends"]
    
    equivalences["checks"] = ["checks", "checks on"]
    
    equivalences["takes care of"] = ["takes care of", "pets", "treats","gives medication", "gives anesthetic", "gives injection", "gives medication to", "gives medicine to"]
    
    equivalences["asks"] = ["asks", "questions", "asks about", "asks for help", "asks about behavior",
                            "asks about situation", "asks permission", "asks for advice", "asks about health","asks", "ask", "questions", "asks about","asks about behavior","asks if he drank",
                            "asks about situation", "asks permission", "asks for advice","asks about health","asks from", "asks again", "asks why", "asks questions", "asks about character",
                            "requests","demands","request","asks about work","asks about plan", "asks opinion", "asks about someone", "asks for clarification", "asks about life",
                            "asks for information","asks about well-being","asks about identity","asks about current situation", "asks about political candidate", "ask questions",
                            "asks about accusation","asks about relationship","asks for information about character", "asks where", "asks question", "asks for permission",
                            "asks about family","asks about well being","asks about the car slowing down","asks if they are under arrest", "asks how they found a hitman", 
                            "asks whether to quit","asks when he finishes", "asks how to fake accidents", "asks for help to clean the cocaine", "asks why he was in the car with his friends", 
                            "asks about his gun", "asks to stop", "asks about emotions", "asks about her dad", "asks for her number", "asks for reason", "asks about friend", 
                            "asks about location", "asks about rape", "asks about orders", "asks about self", "asks about thoughts", "asks about the food", "requests help", 
                            "asks an opinion", "asks talkwho wants to be first", "asks for help from", "asks assistance from", "asks for opinion", "asks to leave", "asks for order", 
                            "asks for money", "asks help from", "asks \xa0for advice", "requests from", "requests 1 minute", "asks brother for a dance", "asks if there is no man with an opinion", 
                            "asks who wants to be first", "asks how to get the stone"]
    
    equivalences["informs"] = ["signals", "reports", "informs","informs about work", "signals", "reports","informs of situation","inform",
                               "informs of rules","informs of news","informs news","informs of current situation", "informs of mistake",
                               "gives report to","signals to","reports to","infforms","infroms","gives info", "inofrms", "informs of political candidate",
                                "shares info","gives information to","informs of plans", "informs viewers", "informs about relationship", 
                                "informs about plan", "gives information", "informs of accident", "informs of plan"]
    
    equivalences["watches (something/someone/with)"] = ["watches", "looks at", "sees", "see","glances at","gapes at", "stares at", "watches from window",
                                        "watches the opera", "watches tv","watch", "glares at","stare at","look at", "watch together","observes",
                                        "watches a game","watch the game","watch the competition", "watch the opera", "watches TV with", 
                                        "watches him choke", "watches leaving", "watch TV together", "exchange looks", "look at each other", 
                                        "look at laptops", "watches them leave", "peeks at", "watches TV", "looking at", "stares", "watches performance", 
                                        "watches television", "watches the pilot", "watches them lie"] 
    
    equivalences["talks (to/with)"] = ["talks to","speaks to","tells","chats","talks about relationships","small talk", "talks incoherently",
                                       "chitchats", "gives speech", "gives speech to", "converses with", "says", "whispers", "talks briefly",
                                       "whispers to","chat","chats with","talk to","talks about work","talks about self","talks", "engage in small talk", 
                                       "engage in a conversation with", "gives advice to", "gives opinion about someone", "tells to take a rest", 
                                       "tells her", "tells accomplishment", "gives a speech to", "calls him a lunatic", "talks to herself", 
                                       "talks to his toys", "talks to God", "talks to himself", "talks to support group", "talks on the speakerphone", 
                                       "speaks to himself", "says he is tired", "gives speech about his alcoholism", "talks to control", 
                                       "says to hit him", "says he bets a nickel", "gives her bridal speech", "gives a speech", "practices his speech", 
                                       "talks to audience", "talks to casino manager", "talks to a friend", "talks to customers", "comments to", 
                                       "gives advice"]
    
    equivalences["talks about (with someone)"] = ["talks about", "talks about work", "talk about","talk about problem", "comments about",
                                   "talks about relationships", "talks about self", "mentions","talk about work","talks about past experiences","talks about men","talks about women",
                                   "talk about plans","talks about relationship","talks about work with","talk about their kids", "chats with people",
                                   "talk about bets","talk","talk about opera","talks about plan","talks about someone", "talks about past",
                                   "talks about concern","talks about meeting later","talks about behavior","talk about plan", "chats with co-workers",
                                   "talks about current situation","talks about work with", "talks about celebrities", "talks about boobs"]  
    
    equivalences["hears/listens"] = ["hears", "listens to","listens to", "listens", "listen to", "hears a noise", "listen", "listens to boss"]
    
    equivalences["shows (something) (to)"] = ["shows", "shows video", "show", "shows scar", "shows trophy to", "shows to", "shows them", "shows hidden alcohol", "shows pill in his mouth", 
                                              "shows him staged pictures", "shows map to", "shows evidence", "shows notebook", "shows computers to colleague"]
    
    equivalences["jokes (with)"] = ["jokes","jokes with","makes funny talk", "jokes about work","joke","makes jokes with", 
                                    "tells a joke to", "joke with", "jokes about"]
    
    equivalences["calls (someone)"] = ["calls", "phone call", "call again", "calls to", "calls on video", "calls him", "calls her", "contacts", "makes pone call", "calls 911 emergency response", 
                                       "calls 911", "makes phone call", "calls superior on the phone", "calls the airline"]
    
    equivalences["ignores"] = ["ignores", "avoids", "tries to avoid", "avoids question"]   
    
    equivalences["introduces (someone/to)"] = ["introduces", "introduces self", "introduces self to", "introduces himself to", 
                                               "introduce","introduce them to each other","introduces himself", "introduces to", 
                                               "introduces him to", "introduces her", "introduces herself", "introduces to others", 
                                               "introduces someone to", "introduces puppy", "introduces herself to", 
                                               "introduces Sonic Death Monkeys"]   
    
    equivalences["works with"] = ["works with","work with"]	
    
    equivalences["reveals (to)"] = ["reveals", "confides in", "opens up", "open up", "reveals to", "opens up to"]
    
    equivalences["talks on the phone (with)"] = ["talks on the phone", "calls on phone","hangs up","talk on the phone","talk on phone",
                                          "talks on phone", "talks on phone with", "on phone", "phones with"]
    
    equivalences["laughs (at/with)"] = ["laughs", "laughs at", "laughing at website ideas", "laugh", "laugh at", "giggles",
                                        "laughs with","laughs together", "laugh together", "giggles with", "laugh with",
                                        "chuckles at"]
    
    equivalences["smiles (at)"] = ["smiles at", "smiles", "smirks at", "smirks"]
    
    equivalences["reads (from/about)"] = ["reads", "reads to", "read the book", "read a book", "reads a letter from", 
                                          "reads note from", "reads about", "reads the report", "read to", "reads the e-mail", 
                                          "reads letter from", "reads a book", "reads paper", "reads speech", "reads newspaper", 
                                          "reads book", "reads planner", "reads mail", "reads notes about a patient", 
                                          "reads notes about the boy", "reads about AIDS", "reads newspapers", "reads note", 
                                          "reads labels", "reads a psychology book", "reads a children's book", 
                                          "reads the notebook", "reads a newspaper", "reads the letter", "reads the diary", 
                                          "reads the book", "reads the note", "reads the paper", "reads the note", 
                                          "reads the question"]
    
    equivalences["looks for/searches"] = ["for", "looks for", "searches", "search for", "searches for", "search the car", 
                                          "search for their other friend", "look for", "searches with", "looks inside the car", 
                                          "searches for the gun", "looking for"]
    
    equivalences["discusses (with)"] = ["discusses", "negotiates", "negotiates with",
                                        "discusses problem", "discuss","discusses with","discuses with", 
                                        "discuss problem", "discuss ideas", "discusses work", 
                                        "discuss work", "discusses reservations", "discuss a problem"] 
    
    equivalences["notices"] = ["notices", "recognizes", "understands", "spots", "notices that his pants are on fire"]   
    
    equivalences["remembers"] = ["remembers", "knows about", "remembers his wife", "remembers wife"]  
    
    equivalences["shushes/calms down"] = ["shushes","hushes", "calms", "tries to calm", "silences", "calms down", "tries to calm down", "calm him down", "calms her", 
                                          "calming down"]  
    
    equivalences["sings (for/with)"] = ["sings to","signs for", "sings", "sings to the cat", "sings with", "stands and sings", "sing with", 
                                        "sings for", "sing a song"]
    
    equivalences["thinks (of/about)"] = ["thinks", "thinks of", "thinking of", "believes in", "assumes", "wonders", "imagines", 
                                         "thinks about her friend", "thinks about his girlfriend", "wonder", "wonders at"]   
    
    equivalences["states"] = ["states", "pretends", "pronounces", "concludes"]  
    
    equivalences["repeats"] = ["repeats", "repeats number 38", "repeats the question", "repeats that he shouldn't have said that", "repeats his fake name"]
    
    equivalences["obeys"] = ["obeys"]
    
    equivalences["gossips"] = ["gossips", "gossips with"]
    
    equivalences["photographs"] = ["photographs", "takes picture of","takes pictures of","photograph","takes photo of", 
                                   "gossips about", "films", "takes photos"]
    
    equivalences["restrains"] = ["restrains", "restrains.", "tries to stop", "stops", "arrests", "stops him"] 
    
    equivalences["argues (with)"] = ["argues", "confronts","argue", "argues awith", "argues ","argues with","argue with","argue about relationship"] 
    
    equivalences["yells (at)"] = ["yells", "screams", "screams at", "scolds", "snaps at", "yells at","scream","yells insults",
                                  "scream and run", "yell questions at", "chides", "shouts at", "shouts for","starts to yell", 
                                  "yells at the cat", "yells at doctors", "yelling", "swears and screams", "screaming", 
                                  "scream and shout", "stands and yells", "screams he doesn't want to hear it", "screams for",
                                  "screams that he is tired and it is all baloney", "start to scream and panic", 
                                  "screams hysterically", "screams and yells", "yells at visitor", "yells at the herd", 
                                  "yells to get down", "yells that she knew it"]   
    
    equivalences["disagrees (with)"] = ["disagrees with", "protests", "disagree with", "disagrees", "disagree",
                                      "criticizes", "interrupts", "dislikes", "disbelieves", "disapproves of"]
    
    equivalences["complains (to/on)"] = ["complains", "complains on", "complains about", "regrets","complains to", "complains about shoes"] 
    
    equivalences["accuses (someone)"] = ["accuses", "blames", "reprimands", "reproaches","accuses her of raping him", "blames her", 
                                         "blames the dog"]
    
    equivalences["insults"] = ["insults","offends", "scoffs at","swears back at", "disrespects","insult","swears","swears at","curses", 
                               "mumbles insults"]
    
    equivalences["teases/bullies/intimidates"] = ["teases", "mocks", "makes fun of", "nags", "bothers", "shames", "spits on", "tries to intimidate",
                                      "ridicules", "taunts", "ridicules someone", "bullies", "snides at", "making fun of", "intimidates"]
    
    equivalences["threatens"] = ["threatens","threatens with a gun","blackmails","interrogates", "threatens to fire him"]
    
    equivalences["refuses"] = ["refuses", "declines","refuses to answer", "refuses food","refuses offer","refuses invite",
                              "refuses help", "dismisses","refuses to help","denies","disobeys","refuses offer from","declines offer",
                              "refuses the invite","refuse to buy","refuses deal","refuses safe","refuses drink","declines invite",
                              "declines invitation","declines offer from","declines loan","refuses kiss","forbids", "says no to", 
                              "refuses medicine", "rebukes", "rejects", "denies the condition"]
    
    equivalences["lies (to)"] = ["lies", "lies to"]
    
    equivalences["commits crime/offense"] = ["crime", "commits crime","kills", "steals","steals from", "steals a ball", "tries to kill", "abuses"]
    
    equivalences["invalid/irrelevant"] = ["Interaction", "stranger", "Action"]
    
    # Loop through the items
    for _, synonyms in equivalences.items():
        # Loop through the list of synonyms 
        for element in synonyms: 
            # Remove extra spaces
            element = element.strip()
    
    return equivalences 


def get_node_edges_matches():
    """
    

    Returns
    -------
    None.

    """
    
    # Edge type matches from nodes 
    matches = dict()
    matches["Scene/Place"] = "location"
    matches["Scene/Context"] = "circumstance"
    matches["Scene/Character"] = "features"
    matches["Scene/Interaction"] = "has"
    matches["Character/Attribute"] = "is"
    matches["Character/Emotion"] = "expresses"
    matches["Interaction/Character"] = "involves"
    matches["Character/Relationship"] = "linked_to"
    
    
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
    
    return matches, edge_matches


#test = get_kukleva_merged_interactions()
#print(list(test.items())[43])