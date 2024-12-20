import csv
import os
import string
import numpy
import pandas as pd
import nltk
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from sklearn.model_selection import train_test_split

model = None
X_train, X_test, y_train, y_test = None, None, None, None
vectorizer = None

# Загрузка датасета
cur_path = os.path.dirname(__file__)
dataset_path = os.path.relpath("..\\dataset\\conf_num.csv", cur_path)

data = pd.read_csv(dataset_path, encoding='utf-8', on_bad_lines='warn')

def print_dataset(dataset):
    print("Текст:")
    for i in range(0, len(dataset)):
        print(i + 1, " => ", dataset[i])
    print("================================")


print_dataset(data['text'])

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Preprocess the text
def preprocess_text(text):

    # Удаление спец символов из текста
    text = text.lower()
    spec_chars = string.punctuation + '\xa0«»\t—…'
    text = "".join([ch for ch in text if ch not in spec_chars])

    # Токенизация
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('russian'))
    tokens = [word for word in tokens if word not in stop_words]

    # Лемматизация
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

def prepare_text_for_model(text_data):
    global vectorizer

    # Получаем отредактированный текст (после токенезации и лемматизации)
    # data['processed_text'] = text_data.apply(preprocess_text)

    vectorizer = CountVectorizer()

    # Extract features from the processed text
    # features = vectorizer.fit_transform(data['processed_text'])

    features = vectorizer.fit_transform(text_data)

    # Convert the features to a dense matrix
    features = features.toarray()

    return vectorizer, features


def make_model(text_data=data['text']):
    """
        Создание модели и предсказание результатов

        :param text_data: обработанные данные из датасета (после лематизации и пр.)
        :return: None
    """
    global model
    global X_train, X_test, y_train, y_test
    global vectorizer

    vectorizer, features = prepare_text_for_model(text_data)

    # Define the model architecture
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(len(vectorizer.get_feature_names_out()),)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    data['label'] = data['label'].astype(int)

    # Train the model ###### Тут ломается :(
    model.fit(features, data['label'], epochs=10, batch_size=64, verbose=1)

    # Divide the dataset into test and training sets.
    X_train, X_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.2)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test Loss:', loss)
    print('Test Accuracy:', accuracy)


def predict_model(text_feature):
    global model

    # Тестирование
    pred = model.predict(text_feature)

    print("\n")
    for i in pred:

        if float(i[0] >= 0.51):
            str_pred = "CONF"
        else:
            str_pred = "NORM"

        print("Результат: ", i, " Тип: ", str_pred)


##### АААААААА оно работает!!""!!!! капаеца фыа выва ываф  ф авыафк фыафафв 19.12.2024
def new_old_vectorizer_process(text_data):
    global vectorizer

    processed_text_data = []
    for i in range(0, len(text_data)):
        processed_text_data.append(preprocess_text(text_data[i]))

    print("processed text data: ")
    print(processed_text_data)

    features = vectorizer.transform(processed_text_data)

    # Convert the features to a dense matrix
    features = features.toarray()

    return features


if __name__ == "__main__":

    # Сначала нужно обучить модель
    make_model()

    # Работает!
    test_data = ["ты не видела куда он тогда пошел? мне очень интересно",
                 "слышала о новой классной кофейне в центре города?",
                 "мне кажется, нам нужно серьезно поговорить. когда ты сегодня освободишься?",
                 "прикрепил пару файлов во вложении"]

    # test_data = "ты не видела куда он тогда пошел? мне очень интересно"
    # processed_text = preprocess_text(test_data)
    td = pd.Series(test_data)

    # feature_test = prepare_text_for_model(td)[1]
    feature_test = new_old_vectorizer_process(td)

    predict_model(feature_test)



