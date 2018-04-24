import json
from pprint import pprint
import copy
fname = './data/data_train.json'
data = json.load(open(fname))
#pprint(data)
tasks = []
task = None
sentences = []
for episode in data["episodes"] :
    for scene in episode["scenes"] :
        task = {"C": "", "Q": "", "A": ""}
        for utterance in scene["utterances"]:
            sentence = []
            for i,token in enumerate(utterance["tokens"]):
                sentence += token
                sentence += utterance["speakers"]
                character_entity = utterance["character_entities"][i]
                task["Q"] = []
                task["A"] = []
                for c in character_entity : 
                    if(len(c) == 3):
                        task["C"] = sentence
                        #print("hello" , token[c[0]])
                        task["Q"] = token[c[0]]
                        task["A"] = c[2]
                        #print(task["C"])
                        tasks.append(copy.deepcopy(task))
                        #pprint(tasks[-1])

pprint(tasks[-2])
                
for t in tasks:
    pprint(t)