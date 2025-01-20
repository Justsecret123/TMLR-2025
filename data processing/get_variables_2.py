#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

def get_merged_interactions():
    """
    Sets and returns the list of equivalences/synonyms of any interaction summary label

    Returns
    -------
    equivalences : dict
        The list of equivalences.

    """

    # Initialize the object 
    equivalences = dict()

    # Set the equivalences 
    equivalences["apologizes (to)"] = ["apologizes to",
                                       "apologize","apologizes","apologizing"]

    equivalences["leaves (with)"] = ["leaves","leave together","gets out of the house","leave","leaves with",
                                     "leaves nurse's station","leaves the office","leaves alone"]

    equivalences["kisses (someone)"] = ["kisses", "tries to kiss","kiss","kisses on cheek","kisses on the cheek",
                                        "kisses goodbye","kiss on bench","hugs and kisses"]

    equivalences["walks (with)"] = ["walks", "walks with","walk with","accompanies","walks to","walks down the hallway",
                                    "walks away","walks away from","takes off","walk together","walk the hallway","walk down the hallway",
                                    "walks behind","walks to the car","walks toward","walk past","walks arm in arm with","sit by pool",
                                    "walks upstairs","walk around"]

    equivalences["follows"] = ["follows","follow","follows after"]

    equivalences["hugs"] = ["hugs", "puts arm around","hug",
                            "cuddles","embraces","embrace"]

    equivalences["sits (near/with)"] = ["sits","sits near","sits at table","sits together","sit down","sits with","sit together",
                                   "sit at the table","sit at table","sit in a car","sit in a restaurant","sits close to",
                                   "sit at a table","sit near","sit","sit on couch","sit in restaurant","sitsr near","sit in bar",
                                   "sits next to","sit on bench","sit in kitchen","sit at kitchen","sits down","'sit in a bar"]

    equivalences["passes by"] = ["passes by","passes in the street"]

    equivalences["comes"] = ["comes home","tries to enter the room", "arrives"]

    equivalences["meets"] = ["meets", "bumps into"]

    equivalences["hits/attacks"] = ["hits","slaps","attacks","punches","hurts","kicks","hits in head","hits in face","spite",
                                    "hits on","hits with elbow","hits in arm","hits under table", "strangles","attack",
                                    "kicks with foot","hits with fist","punches back","punches in belly","punches to the floor"]

    equivalences["fights (with)"] = ["fights","wrestles with","fight in ring",
                                     "fights with","fight","wrestle"]

    equivalences["approaches"] = ["approaches","draws near to","approach","approaches his bed"]

    equivalences["stays"] = ["stays","stands near","stand near","stays near","stays with",
                             "stands on rooftop","waits for","waits for the medication"]
    
    equivalences["spends time (with)"] = ["watches TV with","spends time with","have dinner"]

    equivalences["plays (with)"] = ["plays","plays with","plays cards","plays poker","play tennis","start to play","play poker",
                                    "plays basketball with","play a game","plays ping pong","play cards"]

    equivalences["wakes up"] = ["wakes up","awakens","wake up"]

    equivalences["runs (with)"] = ["runs","runs to","run","run together","start to run",
                            "runs with","rushes","chases","runs after","chase"]

    equivalences["runs away/escapes from"] = ["runs from","runs away","run away","runs away from","escapes from"]

    equivalences["rides/drives"] = ["rides","rides with",
                                    "drives","picks up","rides together","moves van"]

    equivalences["holds (someone)"] = ["holds","holds in arms","holds hands","catches","holds hand","hold hands","holds his arm"]

    equivalences["visits (someone/somewhere)"] = ["visits","visit an office","visit","comes to see"]

    equivalences["dances (with)"] = ["dances with","dance", "dance with","dances"]

    equivalences["pushes"] = ["pushes","touches",
                              "pushes away","pushes to go to the room"]

    equivalences["shoots (at)"] = ["shoots","shoots at","shoot a "]

    equivalences["eats with"] = ["eats with","drinks with","feeds","has dinner"]

    equivalences["takes"] = ["takes","grabs","buys from","takes from","takes away"]

    equivalences["opens"] = ["opens","opens the door for","opens door","opens door for"]

    equivalences["brings"] = ["brings","carries","serves"]

    equivalences["points at"] = ["points at"]

    equivalences["hides from"] = ["hides from"]

    equivalences["tries"] = ["tries"]

    equivalences["puts"] = ["puts"]

    equivalences["throws"] = ["throws","throws at"]

    equivalences["pulls"] = ["pulls	", "drags", "pull"]

    equivalences["sends away"] = ["sends away"]		

    equivalences["has sex (with)"] = ["has sex","has sex with","sex", "makes love to"]		

    equivalences["gives (something)"] = ["gives", "gives to", "pays", "gives a gift","gives gift to",
                                         "gives phone to","gives pen to","gives cigarettes to","gives the ball","gives cookies to",
                                         "gives money to", "gives phone", "sells to", 
                                         "gives a coat to","give to","give somthing", "gives a packet to","gives medical forms to",
                                         "gives the key","gives the lamp","gives the wand","gives another wand",
                                         "gives a bat","gives card to","gives keys to","gives briefcase to","gives coffee to",
                                         "gives the ball back","gives him gums","gives the bag","gives a bag","gives letters to",
                                         "gives letter","gives drawing to","gives device","gives drink to","gives coat to","gives file to"
                                         ]
    

    equivalences["explains (something) (to)"] = ["explains","explains to","expalins to", "instructs","describes to","gives instructions",
                                                 "explains situation","excuses self","teaches", "describes","explains about situation",
                                                 "explains about past experience","explains self","explains about current situation"] 

    equivalences["answers (to)"] = ["answers","responds","replies","responds to","replies to","replies about rape",
                                    "answer","answers about plan","answers about self","asnwers"]

    equivalences["begs"] = ["begs"]

    equivalences["greets"] = ["greets","welcomes","shakes hands","waves at","says goodbye", "shake hands", "shake hands with",
                              "waves to","waves head to","greet","waves goodbye","says goodbye to","shakes hand","salutes",
                              "says goodnight to","says bye to"]

    equivalences["reassures"] = ["reassures","supports","comforts","consoles"]    

    equivalences["suggests (to)"] = ["suggests","advises","warns", "suggest to", "suggests solution",
                                     "gives opinion","proposes", "corrects","asks out","calls out to", 
                                     "gives suggestions","suggests\xa0 to","suggests to"]
    
    equivalences["offers (something) (to)"] = ["offers food to","offers","offers to help","offers to","invites","offers sandwiches to","offers drink to",
                                               "offers coffee to","offer","offers excuses","offers money to","offers to pay","offers a drink",
                                               "offers a ride","offers to buy a drink","offers a seat to","offers ice cream","offers flowers to",
                                               "offers to pick him up","offers water to","offers deal to","offers alcohol to","offers money"]

    equivalences["compliments"] = ["compliments","praises","reciprocates compliment","compliments hands",
                                   "flatters","admires","sympathizes with"]
    
    equivalences["thank (someone)"] = ["thanks","thanks for the help"]

    equivalences["seduces"] = ["flirts with","flirting","flirts","flirt","seduced him","strips for","shows breasts to"]

    equivalences["helps"] = ["helps","assists","saves","defends"]

    equivalences["encourages"] = ["encourages","motivates","wishes luck"]

    equivalences["reminds"] = ["reminds"]

    equivalences["convinces"] = ["convinces","persuades",
                                 "urges","insists","tries to convince","calls for"]

    equivalences["assures/promises (to)"] = ["assures","promises"," swears","promises to find a hitman","promises to","swears to"]
    
    equivalences["congratulates"] = ["congratulates","applauds", "applaud", "applaud for","applauds for","applaud to",
                                     "cheers","cheers for","celebrates","applauds to","toasts to","claps for", 
                                     "cheer for", "cheer on", "cheer","cheer at","awards"]   

    equivalences["leads"] = ["leads","guides","directs"]

    equivalences["agrees (to)"] = ["agrees","confirms to","agrees to","agrees with comment",
                                   "confirms","accepts","allows", "gives permission","agrees","acceptsrequest","agree","approves"]

    equivalences["agrees with"] = ["agrees with","agree with"]

    equivalences["admits"] = ["admits","admits to","confesses to"]

    equivalences["announces"] = ["announces",
                                 "breaks news to", "declares love to"]

    equivalences["receives"] = ["news from",
                                "receives news from", "receives news about"]

    equivalences["wishes"] = ["wishes","hopes for"]

    equivalences["checks"] = ["checks","checks on"]

    equivalences["takes care of"] = ["takes care of", "pets", "treats","gives medication","gives anesthetic","gives injection","gives medication to"]

    equivalences["gives orders (to)"] = ["orders about work","orders","orders to fire","gives orders to","gives orders","orders by microphone",""]

    equivalences["asks (something (to))"] = ["asks", "ask", "questions", "asks about","asks about behavior","asks if he drank",
                                        "asks about situation", "asks permission", "asks for advice","asks about health","asks from",
                                        "requests","demands","request","asks about work","asks about plan", "asks opinion",
                                        "asks for information","asks about well-being","asks about identity","asks about current situation",
                                        "asks about accusation","asks about relationship","asks for information about character",
                                        "asks about family","asks about well being","asks about the car slowing down","asks if they are under arrest"]
    
    equivalences["asks for help"] = ["asks for help to clean the cocaine","asks for help",
                                     "asks for help from","asks help from","requests help",
                                     "asks \xa0for advice"]

    equivalences["informs (about something)"] = ["informs","informs about work", "signals", "reports","informs of situation","inform",
                                                 "informs of rules","informs of news","informs news","informs of current situation",
                                                 "gives report to","signals to","reports to","infforms","infroms","gives info",
                                                 "shares info","gives information to","informs of plans"]

    equivalences["watches"] = ["watches", "looks at", "sees", "see","glances at","gapes at", "stares at", "watches the opera", 
                               "watches tv","watch", "glares at","stare at","look at"]

    equivalences["talks (to/with)"] = ["talks to","speaks to","tells","chats","talks about relationships","small talk",
                                  "chitchats", "gives speech", "gives speech to", "converses with", "says", "whispers", "talks briefly",
                                  "whispers to","chat","chats with","talk to","talks about work","talks about self","talks",
                                  "talk about work","talks about past experiences","talks about men","talks about women",
                                  "talk about plans","talks about relationship","talks about work with","talk about their kids",
                                  "talk about bets","talk","talk about opera","talks about plan","talks about someone",
                                  "talks about concern","talks about meeting later","talks about behavior","talk about plan",
                                  "talks about current situation","talks about work with", "catches up with"]

    equivalences["talks about"] = ["talks about",
                                   "talk about","talk about","mentions"]

    equivalences["hears"] = ["hears"]

    equivalences["listens"] = ["listens to", "listens", "listen to"]

    equivalences["shows"] = ["shows"]

    equivalences["jokes (with)"] = ["jokes","jokes with","makes funny talk",
                                    "jokes about work","joke","makes jokes with"]

    equivalences["calls"] = ["calls","phone call","call again","calls to"]

    equivalences["ignores"] = ["ignores","avoids","tries to avoid"]

    equivalences["introduces"] = ["introduces",
                                  "introduces self","introduces to","introduce"]

    equivalences["works with"] = ["works with"]

    equivalences["reveals (to)"] = ["reveals", "confides in","opens up", "open up", "reveals to"]

    equivalences["talks on the phone (with)"] = ["talks on the phone", "calls on phone","hangs up","talk on the phone","talk on phone",
                                          "talks on phone","talks on phone with","on phone"]

    equivalences["laughs (at)"] = ["laughs", "laughs at","laughing at website ideas",
                                   "laugh", "laugh at","giggles"]
    
    equivalences["laughs with"] = ["laughs with","laughs together",
                                   "laugh together","giggles with","laugh with"]

    equivalences["smiles (at)"] = ["smiles at","smirks at","smile","looks at and smiles","smiles"]

    equivalences["reads (from/about)"] = ["reads", "reads to","reads note from","reads about","reads the report","read the book","read a book","reads a letter from"]

    equivalences["looks for"] = ["looks for", "searches"]

    equivalences["discusses (with)"] = ["discusses", "negotiates", "negotiates with",
                                        "discusses problem", "discuss","discusses with","discuses with"]

    equivalences["notices"] = ["notices", "recognizes", "understands", "spots"]

    equivalences["remembers"] = ["remembers","knows about"]

    equivalences["shushes/calms down"] = ["shushes","hushes",
                                          "calms","tries to calm","silences","calms down"]

    equivalences["sings to"] = ["sings to"]

    equivalences["thinks (of)"] = ["thinks", "thinks of",
                                   "believes in","assumes","wonders","thinking of"]

    equivalences["states"] = ["states", "pretends", "pronounces","concludes"]

    equivalences["repeats"] = ["repeats"]

    equivalences["obeys"] = ["obeys"]

    equivalences["gossips"] = ["gossips", "gossips with"]

    equivalences["photographs (someone)"] = ["photographs", "takes picture of","takes pictures of","photograph","takes photo of"]

    equivalences["restrains"] = ["restrains","tries to stop","stops"]

    equivalences["argues (with)"] = ["argues", "confronts","argue",
                                     "argues ","argues with","argue with","argue about relationship"]

    equivalences["yells/screams (at)"] = ["yells", "screams","scolds",
                                          "snaps at", "yells at","scream","scream and run","yell questions at"]

    equivalences["disagrees (with)"] = ["disagrees with", "disagrees", "protests","disagree with",
                                        "criticizes", "interrupts", "dislikes", 
                                        "disbelieves", "disapproves of", 
                                        "says no to","disagrees","disagree"]

    equivalences["complains"] = ["complains",
                                 "complains about", "regrets","complains to"]

    equivalences["accuses"] = ["accuses", "blames",
                               "reprimands", "reproaches","accuses her of raping him"]

    equivalences["insults"] = ["insults","offends", "scoffs at","swears back at",
                               "disrespects","insult","swears","swears at","curses"]

    equivalences["teases/ridicules (someone)"] = ["teases", "mocks","makes fun of","nags","bothers","shames","ridicules","taunts"]

    equivalences["threatens/pressures"] = ["threatens","threatens with a gun","points gun","points a gun at","blackmails","interrogates",
                                 "threatens to fire him","points gun at"]

    equivalences["refuses (to do) / something (from)"] = ["refuses", "declines","refuses to answer","refuses food","refuses offer","refuses invite",
                                                "refuses help", "dismisses","refuses to help","denies","disobeys","refuses offer from","declines offer",
                                                "refuses the invite","refuse to buy","refuses deal","refuses safe","refuses drink","declines invite",
                                                "declines invitation","declines offer from","declines loan","refuses kiss","forbids"]

    equivalences["lies"] = ["lies","lies to"]

    equivalences["commits (crime)"] = ["crime", "commits crime","kills", "steals","steals from"]

    equivalences["kneels in front of"] = ["kneels in front of"]
    
    equivalences["does something for"] = ["plays music for","dances for","holds door for","lights a cigarette for","opens car door for",
                                          "pours drink for","opens the doors for","brings food for","sings for"]

    equivalences["invalid/irrelevant"] = ["Interaction","why don't they go little close","mistakes"]


    # Loop through the items
    for _, synonyms in equivalences.items():
        # Loop through the list of synonyms 
        for element in synonyms: 
            # Remove extra spaces
            element = element.strip()

    
    return equivalences