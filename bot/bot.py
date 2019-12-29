import numpy
import tflearn
import tensorflow as tf
import random

from bot.src.lib.important import bag_of_words, extract_from_json, train_to_bow

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Chatbot class for production
class Chatbot:
    def __init__(self, name):
        self.name = name
        self.words, self.labels, self.train_x, self.train_y = ([] for i in range(4))

    def get_response(self, input_statement):
        self.input_statement = input_statement
        predictions = self.model.predict([bag_of_words(self.input_statement, self.words)])[0]
        index = numpy.argmax(predictions)
        tag = self.labels[index]
        if predictions[index] > 0.7:
            for tg in self.data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            response = random.choice(responses)
        else:
          response = 'Maaf kami tidak dapat memproses pertanyaan anda. Silahkan ganti pertanyaan lain.'

        return response
    
    def set_training(self, data):
        # Extract data from json
        self.data = data
        self.words, self.labels, self.train_x, self.train_y = extract_from_json(self.data)

        self.words = [stemmer.stem(w.lower()) for w in self.words if w != "?"]
        self.words = sorted(list(set(self.words)))
        self.labels = sorted(self.labels)

        # Transform data into bag of words
        training, output = train_to_bow(self.words, self.labels, self.train_x, self.train_y)

        # Training Model
        tf.reset_default_graph()
        net = tflearn.input_data(shape=[None, len(training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
        net = tflearn.regression(net)

        self.model = tflearn.DNN(net)
        self.model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        self.model.save("bot/model/model.tflearn")
