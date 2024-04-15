# import necessary libraries

import json
import random
import pickle
import numpy as np
import tensorflow 
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

lemmatizer = WordNetLemmatizer()

# load intents from json file
intents = json.loads(open(r"C:\Users\user\Medical Chatbot\Medical-Chatbot\Mental-Health-Chatbot_intents").read())

# create empty lists for words, classes and documents
words = []
classes = []
documents = []
ignore_letters = ["?", ",", ".", "!", ":", ";", "'", '"', "-", "_", "(", ")", "[", "]", "{", "}", "/", "\\"]

# loop through each sentence in the intents patterns and tokenize the words
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# lemmatize and lowercase each word, remove duplicates, and sort alphabetically
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))

# sort classes alphabetically
classes = sorted(list(set(classes)))

# save words and classes as pickle files
pickle.dump(words, open("Mental-Health_data_words.pkl", "wb"))
pickle.dump(classes, open("Mental-Health_data_classes.pkl", "wb"))

# create training data
training = []
output_empty = [0] * len(classes)

# loop through each document and create a bag of words and output row
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# shuffle training data and convert to numpy array
random.shuffle(training)
training = np.array(training)

# split training data into input (X) and output (y)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# create the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu", kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu", kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# use the Adam optimizer and a learning rate scheduler to adjust learning rate during training
opt = Adam(learning_rate=0.001)

def lr_scheduler(epoch):
    return 0.001 * pow(0.1, epoch//10)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# fit the model
batch_sizes = [5, 10, 15, 20, 25]
epochs = [50, 100, 150, 200]

for batch_size in batch_sizes:
    for epoch in epochs:
        hist = model.fit(np.array(train_x), np.array(train_y), epochs=epoch, batch_size=batch_size, verbose=1, callbacks=[tensorflow.keras.callbacks.LearningRateScheduler(lr_scheduler)])
# save the model to disk
model.save("Mental-Health-Chatbot_model.h5", hist)
print("Model created and saved to disk")