import numpy
import nltk
nltk.download('punkt')

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def bag_of_words(s, words):
    '''
    Parameter:
        s    : input value
        words: list of training stemmed words
    return:
        array of vectorized input
    '''
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def extract_from_json(data):
    '''
    Parameter:
        - data: json from intents.json
    return:
        - words  : list of words
        - labels : list of intents labels
        - train_x: train data of predictor
        - train_y: train data of response
    '''
    words, labels, train_x, train_y = ([] for i in range(4))
    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            train_x.append(wrds)
            train_y.append(intent["tag"])
            
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    return words, labels, train_x, train_y

def train_to_bow(words, labels, train_x, train_y):
    '''
    Parameter:
        - words  : list of words
        - labels : list of intents labels
        - train_x: train data of predictor
        - train_y: train data of response
    return:
        - training: list of vectorized predictor
        - output  : list of vectorized response
    '''
    training, output = ([] for i in range(2))
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(train_x):
        bag = []
        stemmed_words = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in stemmed_words:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(train_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)
    return training, output
