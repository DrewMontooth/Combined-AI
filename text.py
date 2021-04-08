import json
import random
import re
from typing import Dict, List, Text

import numpy as np
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
import pandas as pd

# Name
# Progam Name
# Age
# Platform
# Language

print("_"*70)


class MarkovChain:
    def __init__(self, text):
        self.corpus = text.split("\n")

        self.words = ' '.join(self.corpus)
        self.words = re.split(r'[\.|\,|\?|\;|\!| ]', self.words)

    def make_graph(self):
        self.graph = {}
        self.mark = 1

        for word in range(len(self.words)):
            if self.mark != len(self.words):
                if self.words[word] in self.graph:
                    self.graph[self.words[word]].append(self.words[word + 1])
                else:
                    self.graph[self.words[word]] = [self.words[word + 1]]
            else:
                self.graph[self.words[word]] = [self.words[0]]

            self.mark += 1


    def generate_text(self, seed=Text, words_generated=100):
        self.chain = seed.split()

        for _ in range(words_generated):
            try:
                self.chain.append(random.choice(self.graph[self.chain[-1]]))
            except IndexError:
                print('Chain Broken')
                break

        return ' '.join(self.chain)


# class Encode:
#     @staticmethod
#     def str2bin(sentences):
#         """Creates ids for words in sentence"""
#         sents = []

#         for sentence in sentences:
#             spilt_sentence = sentence.strip(" ")
#             utf8_array = [word.encode("utf-8") for word in spilt_sentence]
#             word_ids = []
#             for word in utf8_array:
#                 for character in word:
#                     word_ids.append(character)

#             binary = ""
#             for word_id in word_ids:
#                 binary += "{0:b}".format(word_id)

#             string_list = []
#             for stringval in binary:
#                 string_list.append(int(stringval))
#             sents.append(string_list)

#         return sents

#     @staticmethod
#     def pad(sents):
#         padded = np.zeros([len(sents), 10000])
#         for i, j in enumerate(sents):
#             padded[i][0:len(j)] = j

#         return padded


# class SentimentAnalysis:
#     def __init__(self, dataset):
#         self.dataset = dataset

#     def split_data(self):
#         self.trainX = []
#         self.trainY = []
#         self.testX = []
#         self.testY = []

#         for i in self.dataset["data"]:
#             self.trainX.append(i['text'])
#             self.trainY.append(i['label'])

#         for i in self.dataset['test']:
#             self.testX.append(i['text'])
#             self.testY.append(i['label'])

#     def get_context(self, pretty_print=False):
#         train_data = Encode.str2bin(self.trainX)
#         outputs = np.array([self.trainY]).T
#         test_data = Encode.str2bin(self.testX)
#         test_labels = self.testY
#         padded = Encode.pad(train_data)
#         testing = Encode.pad(test_data)

#         np.random.seed(1)
#         def sigmoid(x): return 1 / (1 + np.exp(-x))
#         def sigmoid_diriv(x): return x * (1 - x)

#         weights = 2 * np.random.random((padded.shape[1], 1)) - 1

#         for _ in range(20000):
#             output = sigmoid(np.dot(padded, weights) - 1)
#             error = outputs - output
#             adjustment = np.dot(padded.T, error * sigmoid_diriv(output))
#             weights += adjustment

#         output = sigmoid(np.dot(testing, weights) - 1)

#         if pretty_print != False:
#             label_index = 0

#             for value in np.round(output):
#                 if label_index != len(test_labels) - 1:
#                     if value == 1:
#                         print(f'AI guess: {value}')
#                         print(f'Actual: {test_labels[label_index]}')
#                         if value == test_labels[label_index]:
#                             print('Estimation is accuracte')
#                         else:
#                             print('Estimation is not accurate')
#                     else:
#                         print(f'AI guess: {value}')
#                         print(f'Actual: {test_labels[label_index]}')
#                         if value == test_labels[label_index]:
#                             print('Estimation is accuracte')
#                         else:
#                             print('Estimation is not accurate')
#                     print()

#                 elif label_index == len(test_labels) - 1:
#                     if value == 1:
#                         print(f'AI guess: {value}')
#                         print(f'Actual: {test_labels[label_index]}')
#                         if value == test_labels[label_index]:
#                             print('Estimation is accuracte')
#                         else:
#                             print('Estimation is not accurate')
#                     else:
#                         print(f'AI guess: {value}')
#                         print(f'Actual: {test_labels[label_index]}')
#                         if value == test_labels[label_index]:
#                             print('Estimation is accuracte')
#                         else:
#                             print('Estimation is not accurate')

#                 label_index += 1
#         else:
#             print('Finished!')
#             print(f'Unrounded Output: {output}')
#             print(f'Rounded Output: {np.round(output)}')

#         if np.round(output) == 1:
#             context = 'good'
#         else:
#             context = 'bad'

#         return context


def get_pos(text):
    pos = []
    POS = pos_tag(word_tokenize(text))
    for i in POS:
        pos.append(i[-1])

    return pos


def split_by(text, by):
    n = text.split(" ")
    key_index = []
    i = 0
    for x in n:
        if x in by:
            key_index.append(i)
        i += 1

    sent = []
    str2 = ''
    for k in range(len(n)):
        if k in key_index and k != 0:
            str2 += '. '

        str2 += n[k]
        str2 += ' '
    return str2.split(" . ")


def compare(base_text, sent):
    lines = base_text.split("\n")
    starting = []
    middle = []
    ending = []
    t = None
    for line in lines:
        words = line.split(" ")
        if sent.split() == words:
            return None
        else:
            starting.append(words[0])
            ending.append(words[-1])
            if len(words) < 3:
                middle.append(words[1:])
            else:
                middle.append(words[1:-1])

    starting = list(set(starting))
    ending = list(set(ending))

    split_sentence = sent.split()
    for x in split_sentence:
        if x == '' or x == ' ':
            split_sentence.remove(split_sentence[split_sentence.index(x)])
    sent_start, sent_middle, sent_end = (
        split_sentence[0], split_sentence[1:-1], split_sentence[-1])
    # print(sent_start, sent_middle, sent_end)
    # print(starting, ending)
    if sent_start in starting and sent_end in ending:
        # print('Passed!')
        for i in middle:
            # double inner check system
            # print(f'{i}{" "*(65-len(str(i)))}{sent_middle}')
            # print(f'{[i[0]]}{" "*(65-len(str([i[0]])))}{sent_middle}')
            # print(f'{[i[-1]]}{" "*(65-len(str([i[-1]])))}{sent_middle}')
            if sent_middle == i:
                return None
            else:
                if i[0] == sent_middle[0] or i[-1] == sent_middle[-1]:
                    return sent
    else:
        return None
    # if sent_start in starting and sent_end in ending:
    #     for i in middle:
    #         if len(split_sentence) > 4:
    #             together = ' '.join(i[-3:])
    #             together_sent = ' '.join(sent_middle[-3:])
    #         else:
    #             together = ' '.join(i[-2:])
    #             together_sent = ' '.join(sent_middle[-2:])
    #         if together == together_sent:
    #             return sent
    # else:
    #     return None


def grade(input, text):
    print(text.strip())
    if int(input) == 0:
        with open('NLP/wrong_answers.txt', 'a') as f:
            f.write('\n'+text)
    elif int(input) == 2:
        with open('Smart_AI/base_txt.txt', 'a') as f:
            f.write('\n'+text)


base_text = open('Smart_AI/base_txt.txt', 'r').read()

starting_words = list(set([line.split(" ")[0]
                           for line in base_text.split("\n")]))
# print(starting_words)
seed = random.choice(starting_words)


# def run():
#     words = random.randint(3, 30)

#     markov = MarkovChain(base_text)
#     markov.make_graph()
#     text = markov.generate_text(seed, words)

#     with open('NLP/wrong_answers.txt', 'r') as f:
#         if text in list(f.read().split('\n')):
#             run()

#         else:
#             print()
#             print('Generated Text:', text)
#             print('Words:', words)

#             split_ = split_by(text, starting_words)
#             sent = split_[split_.index(random.choice(split_))]

#             if len(sent.split()) < 3:
#                 run()

#             elif len(sent.split()) >= 3:
#                 split_ = split_by(sent, starting_words)
#                 sent = split_[split_.index(random.choice(split_))]

#                 print(sent)

#                 with open('Smart_AI/base_txt.txt', 'r') as b:
#                     x = b.read()

#                     if sent in list(base_text.split('\n')):
#                         run()

#                     elif sent not in list(x.split('\n')):
#                         accurate = compare(base_text, sent)

#                         if accurate != None:
#                             print("_"*70)
#                             print(f'Fixed Output: {sent}')
#                             return str(accurate).strip()

#                         elif accurate == None:
#                             print("_"*70)
#                             run()


# text = run()
# while text == None:
#     text = run()
# # ask for user input
# approve = input("Is this good (2 to add to base | 1 for good | 0 for bad): ")

# grade(approve, text)