import string

import gensim
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import nltk

tokenizer = RegexpTokenizer(r'\w+')
en_stop = list(stopwords.words('english'))
p_stemmer = WordNetLemmatizer()

def get_topic(sents, topics=2, num_words=1, print_outputs=False):
    # texts = [[word for word in document.lower().split() if word not in list(stopwords.words('english'))] for document in sents]
    # print(texts[0])
    
    texts = []

    # loop through document list
    for i in sents:
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if not i in en_stop]
        stemmed_tokens = [p_stemmer.lemmatize(i) for i in stopped_tokens]
        texts.append(stemmed_tokens)

    dictionary = gensim.corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    # print(corpus)

    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, 
                                          num_topics=topics, 
                                          id2word=dictionary, 
                                          passes=1000,
                                          alpha='auto',
                                          minimum_probability=0.0,
                                          random_state=1)
    lda_topics = lda.show_topics(formatted=False, num_words=num_words, num_topics=topics)

    if num_words > 1:
        topicz = []
        for idx, topic in lda_topics:
            if print_outputs == True:
                print('Topic: {} \nWords: {}\n'.format(idx, [w[0] for w in topic]))
            topicz.append([w[0] for w in topic])
        return topicz
    else:
        return [w[0] for idx, topic in lda_topics for w in topic]

# docs = ["The sky is blue",
#         "The world is so kind and nice"]

# print(get_topic(docs, 2, 1))