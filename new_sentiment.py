from numpy.core.records import array
from sklearn.neural_network import MLPClassifier
import numpy as np
from typing import List, Tuple


dataset = {
    "data": [
        {"text": 'I dont like you', "label": 0},
        {"text": 'I like you', "label": 1},
        {"text": 'I dont like apples', 'label': 0},
        {"text": 'I like apples', 'label': 1},
        {"text": 'I like bread', 'label': 1},
        {"text": 'like bread I', 'label': 1},
        {"text": 'I dont like oranges', 'label': 0},
        {"text": 'dont like oranges I', 'label': 0},
        {"text": 'I am happy', 'label': 1},
        {"text": "happy am I", 'label': 1},
        {"text": 'I love this', 'label': 1},
        {"text": 'love this I', 'label': 1},
        {"text": 'I did not like this', 'label': 0},
        {"text": 'did not this like I', 'label': 0},
        {"text": 'I didnt like this', 'label': 0},
        {"text": 'didnt this like I', 'label': 0},
        {"text": "I like this very much", 'label': 2},
        {"text": "this like very I much", 'label': 2},
        {"text": "oogabooga", "label": -1},
        {"text": 'I like being good', 'label': 1},
        {"text": 'good being like I', 'label': 1},
        {"text": 'I like anything', 'label': 1},
        {'text': 'anything I like', 'label': 1},
        {'text': 'I like sloths', 'label': 1},
        {"text": 'like sloths I', 'label': 1},
        {"text": 'I like algorithms', 'label': 1},
        {'text': 'like I algorithms', 'label': 1}
    ],
    "test": [
        {"text": 'I like python', 'label': 1},
        {"text": 'I dont like java', 'label': 0},
        {"text": 'I like being happy', 'label': 1},
        {"text": 'I dont like being mad', 'label': 0},
        {"text": 'I like algorithms', 'label': 1},
        {"text": 'I love you', 'label': 1},
        {"text": 'I did not like this meal that much', 'label': 0},
        {'text': 'I love this very much', 'label': 2},
        {'text': 'oogabooga', 'label': -1}
    ]
}


class Encode:
    @staticmethod
    def str2bin(sentences):
        """Creates ids for words in sentence"""
        sents = []
        for sentence in sentences:
            spilt_sentence = sentence.strip(" ")
            utf8_array = [word.encode("utf-8") for word in spilt_sentence]
            word_ids = []
            for word in utf8_array:
                for character in word:
                    word_ids.append(character)

            binary = ""
            for word_id in word_ids:
                binary += "{0:b}".format(word_id)

            string_list = []
            for stringval in binary:
                string_list.append(int(stringval))
            sents.append(string_list)
        return sents

    @staticmethod
    def pad(sents):
        padded = np.zeros([len(sents), 1000])
        for i, j in enumerate(sents):
            padded[i][0:len(j)] = j
        return padded


class SentimentAnalysis:
    def __init__(self, dataset):
        self.dataset = dataset
        self.ran_before = 0
    def split_data(self):
        self.trainX, self.trainY, self.testX, self.testY = ([], [], [], [])

        for i in self.dataset["data"]:
            self.trainX.append(i['text'])
            self.trainY.append(i['label'])

        for i in self.dataset['test']:
            self.testX.append(i['text'])
            self.testY.append(i['label'])

    def get_context(self, context=False, 
                    print_none=True, regular=True, 
                    zero_negative=True, added_test: List[Tuple["Text", "Label"]]=...):
        # context - return the context of the last item in the dataset
        # print_none - print nothing
        # regular - whether to give the whole/regular output
        # zero_negative - whether to subtract when the label is 0
        # added_test - added items to get
        if context == True:
            regular = False

        if self.ran_before == 0:
            self.trainX = Encode.str2bin(self.trainX)
            self.trainY = np.array([self.trainY]).T

            # if added_test != Ellipsis and test_only_new == True:
            #     for i in added_test:
            #         self.testX.append(i[0])
            #         self.testY.append(i[1])
                    
            self.testX = Encode.str2bin(self.testX)
            self.testY = self.testY

            self.trainX = Encode.pad(self.trainX)
            self.testX = Encode.pad(self.testX)
            self.ran_before += 1

        nn = MLPClassifier(random_state=1, max_iter=5000, activation='relu').fit(self.trainX, np.ravel(self.trainY, order='C'))
        
        if added_test != Ellipsis:
            new = []
            for i in added_test:
                new.append(i[0])
            new = Encode.str2bin(new)
            new = Encode.pad(new)
            self.prediction = nn.predict(new)
            return np.array(self.prediction).tolist()

        else:
            self.prediction = nn.predict(self.testX)

        if print_none == False:
            for x in range(len(self.testX)):
                accuracy = sum(1 for x, y in zip([nn.predict([self.testX[x]])[0]], [self.testY[x]]) if x == y) / float(len([nn.predict([self.testX[x]])[0]]))
                print(f'Expected: {self.testY[x]}, Got: {nn.predict([self.testX[x]])[0]}, Accuracy: {accuracy * 100}')

            print(f'Score: {round((nn.score(self.testX, self.testY) * 100), 2)}')

        if regular == True:
            return self.prediction
        else:
            if zero_negative == True:
                if context == True:
                    return int(np.array([self.prediction[-1]])[0])
                else:
                    self.happiness = 0
                    for i in range(len(self.prediction)):
                        if self.testX[i] == self.prediction[i]:
                            if self.prediction[i] > 0:
                                self.happiness += 1
                            else:
                                self.happiness -= 1

                    return self.happiness
            else:
                if context != False:
                    return int(np.array([self.prediction[-1]])[0])
                else:
                    self.happiness = 0
                    for i in range(len(self.prediction)):
                        if self.testX[i] == self.prediction[i]:
                            if self.prediction[i] > 0:
                                self.happiness += 1
                            elif self.prediction[i] < 0:
                                self.happiness -= 1

                    return self.happiness


# a = SentimentAnalysis(dataset)
# a.split_data()
# happiness = a.run(True)

# print(f'Happiness: {happiness}')