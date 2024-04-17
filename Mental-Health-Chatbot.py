import json
import random
import pickle
import numpy as np 
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open(r"C:\Users\user\Projects\Medical Chatbot\Mental-Health-Chatbot_intents").read())
words = pickle.load(open(r"C:\Users\user\Projects\Medical Chatbot\Mental-Health_data_words.pkl", "rb"))
classes = pickle.load(open(r"C:\Users\user\Projects\Medical Chatbot\Mental-Health_data_classes.pkl", "rb"))
model = load_model(r"C:\Users\user\Projects\Medical Chatbot\Mental-Health-Chatbot_model.h5")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag= [0] * len(words)

    for w in sentence_words:
        for i , word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.1
    results = [[i, r] for i ,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort( key=lambda x: x[1], reverse= True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]],"probability": str(r[1 ])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    else:
        result = "I'm sorry, I didn't understand what you just said."
    return result

while True:
    message = input("")
    if len(message) > 0:
        ints = predict_class(message)
        res = get_response(ints, intents)
        print(res)
    else:
        print("Please enter a message.")

