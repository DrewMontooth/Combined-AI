import cv2
import numpy as np

import text; text.__all__ = ["MarkovChain", "get_pos", "compare", "split_by", 
    "grade"]; from text import *
# from summarizing import summarize, summarize_url
import random

from new_sentiment import *
from topic import get_topic


###############################################################################################################
# RTOD
###############################################################################################################

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
conf_threshold = 0.5
nms_threshold = 0.3

classesDir = 'Object_Detection/coco.names'
classNames = []
with open(classesDir, "r") as f:
    classNames = f.read().split("\n")


# model_cfg, model_weights = 'C:/Users/DrewM/OneDrive/Documents/Code/Object_Detection/yolov3.cfg', 'C:/Users/DrewM/OneDrive/Documents/Code/Object_Detection/yolov3.weights'
tiny_cfg, tiny_weights = 'C:/Users/DrewM/OneDrive/Documents/Code/Object_Detection/yolov3-tiny.cfg', 'C:/Users/DrewM/OneDrive/Documents/Code/Object_Detection/yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(tiny_cfg, tiny_weights)

# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

names_places = []
def find_objects(outputs, img, return_=False):
    global name
    hT, wT, channelsT = img.shape
    bbox = []
    class_ids = []
    confs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                w, h = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int((detection[0] * wT) - w/2), int((detection[1] * hT) - h/2)
                bbox.append([x, y, w, h])
                class_ids.append(class_id)
                confs.append(float(confidence))

    indecies = cv2.dnn.NMSBoxes(bbox, confs, conf_threshold, nms_threshold)
    

    for i in indecies:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        names_places.append(classNames[class_ids[i]])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[class_ids[i]]} {int(confs[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


names = []
# create a while loop to capture each frame
while True:
    success, img = cap.read()

    # convert our image to blob format which the NN can input
    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()] # gets the yolo_82, yolo_94, yolo_106 layers which are all output layers


    outputs = net.forward(outputNames)

    find_objects(outputs, img)
    # if len(names_places) >= 3:
    #     # names_places = list(set([i[0] for i in names_places]))
    #     # names.append(names_places)
    #     # names_places = []

    cv2.imshow('Object Detector', img)
    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    # cv2.waitKey(1)

# names = list(set([i[0] for i in names]))
print(names, list(set(names_places)))
names_places = list(set(names_places))
###############################################################################################################
# Thought
###############################################################################################################

base_text = open('Smart_AI/thought_base.txt', 'r').read()

happiness = 0
happiness_dataset = {
    "data": [
        {'text': 'Humans are so weird', 'label': 0},
        {'text': 'weird Humans so are', 'label': 0},
        {'text': 'You are human', 'label': 0},
        {'text': 'human are You', 'label': 0},
        {'text': 'Cats are weird', 'label': 1},
        {'text': 'weird Cats are', 'label': 1},
        {'text': 'Dogs are cute', 'label': -1},
        {'text': 'are Dogs cute', 'label': -1}
    ],
    "test": [
        {'text': names_places[0], 'label': 0}
    ]
}

# print(base_text)
starting_words = list(set([line.split(" ")[0] for line in base_text.split("\n")]))
# print(starting_words)
seed = random.choice(starting_words)

ss = SentimentAnalysis(happiness_dataset)
ss.split_data()

def run():
    global happiness
    words = random.randint(3, 30)

    markov = MarkovChain(base_text)
    markov.make_graph()
    text = markov.generate_text(seed, words)

    split_ = split_by(text, starting_words)
    sent = split_[split_.index(random.choice(split_))]
    if sent == "You're so much" or sent == "You're so much ":
        return None

    with open('NLP/wrong_answers.txt', 'r') as f:
        if sent in list(f.read().split('\n')):
            return None
        else:
            # print('Generated Text:', text)
            # print('Words:', words)

            if len(sent.split()) < 3:
                return None
                
            elif len(sent.split()) >= 3:
                split_ = split_by(sent, starting_words)
                sent = split_[split_.index(random.choice(split_))]

                # print(sent)

                with open('Smart_AI/thought_base.txt', 'r') as b:
                    x = b.read()

                    if sent in list(x.split('\n')):
                        print('Found Base Text Copy!')
                        return None

                    elif sent not in list(x.split('\n')):
                        accurate = compare(base_text, sent)

                        if accurate is not None:
                            # happiness_dataset['test'].append({'text': accurate, 'label': 0})
                            # ss = SentimentAnalysis(happiness_dataset)
                            # ss.split_data()
                            context = ss.get_context(context=True, print_none=True, added_test=[(accurate, 0)])
                            print(f'Output: {sent}')
                            print(f'Context: {context}')
                            print()

                            topic = get_topic([accurate.lower()], 1, 1)
                            topic2 = get_topic([happiness_dataset['test'][0]['text'].lower()], 1, 1)
                            print(topic, topic2)

                            print("_"*70)
                            if topic == topic2:
                                print(f"Output: {str(accurate).strip()}")
                                return str(accurate).strip()
                            else:
                                context2 = ss.get_context(context=True, print_none=True, added_test=[(topic[0], 0), (topic2[0], 0)])
                                # print(context2)
                                if context2[0] == context2[1]:
                                    print(f"Output: {str(accurate).strip()}")
                                    return str(accurate).strip()
                                else:
                                    return None
                            # if context == 1:
                            #     happiness += 0.1
                            # elif context == 0:
                            #     happiness -= 0.1
                            # # print("_"*70)
                            # # print()
                            # print(f'Output: {str(accurate).strip()}')
                            # print(f'Current Happiness: {happiness}')
                            # return str(accurate).strip()

                        elif accurate == None:
                            # print("_"*70)
                            return None

text = run()
while text == None:
    text = run()

# ask for user input
# approve = input("Is this good (2 to add to base | 1 for good | 0 for bad): ")

# grade(approve, text)
