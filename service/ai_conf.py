import datetime
import os
import string
import numpy as np
import pandas as pd
import nltk
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
import tensorflow as tf
from sklearn.model_selection import train_test_split

model = None
X_train, X_test, y_train, y_test = None, None, None, None
vectorizer = None

# Тестовые данные в массивах (если потребуются)
test_data = [
    "ты не видела куда он тогда пошел? мне очень интересно",
    "слышала о новой классной кофейне в центре города?",
    "мне кажется, нам нужно серьезно поговорить. когда ты сегодня освободишься?",
    "прикрепил пару файлов во вложении"
]

hard_test_data = [
    "Я слышал, что на высшем уровне обсуждают изменения в стратегии, но детали держат в секрете.",
    "Некоторые из нас получили запросы на дополнительную информацию, но неясно, для чего она нужна.",
    "Мне сказали, что скоро будут объявления о новых назначениях, но никто не знает, кто именно будет назначен.",
    "В офисе говорят, что есть проблемы с одним из проектов, но официальной информации пока нет.",
    "На последней встрече упоминали о возможных увольнениях, но никто не подтвердил эту информацию."
]

normal_data = [
    "С днем рождения, Ирина! Желаю счастья и успехов во всех начинаниях!",
    "Привет! Как дела? Давно не виделись!",
    "Кто-то хочет пойти в кино в выходные? Напишите, если интересно!",
    "Участвую в марафоне на следующей неделе! Кто со мной?",
    "Заметила, что у нас много общих друзей! Как ты знаешь?"
]

conf_data = [
    "Мне нужно обсудить с тобой некоторые личные и деловые вопросы, касающиеся работы.",
    "Это пока неофициально, так что прошу держать это в секрете.",
    "Привет! У меня есть информация о предстоящем проекте, которую нельзя разглашать",
    "Не хочу тебя пугать, но я слышал слухи о возможных увольнениях в компании.",
    "Я хотел бы обсудить свою зарплату и возможные изменения. Это важный вопрос."
]
# Получение пути датасета (т.к. находится все в другой, внешней папке)
cur_path = os.path.dirname(__file__)
correct_path = os.path.relpath("..\\dataset", cur_path)
conf_dataset = correct_path + "/conf_num.csv"

# Загружаем датасет
data = pd.read_csv(conf_dataset, encoding='utf-8', on_bad_lines='warn')


def print_dataset(dataset):
    print("Текст:")
    for i in range(0, len(dataset)):
        print(i + 1, " => ", dataset[i])
    print("================================")


# Печатаем датасет (для определения возможных ошибок, опционально)
print_dataset(data['text'])

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


# Обрабатываем текст для модели
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

    # Стемминг (спорно, но можно оставить)
    stemmer = SnowballStemmer(language='russian')
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)


def prepare_text_for_model(text_data):
    global vectorizer

    # Получаем отредактированный текст (после токенезации и лемматизации)
    data['processed_text'] = text_data.apply(preprocess_text)

    vectorizer = TfidfVectorizer()

    # Extract features from the processed text
    # features = vectorizer.fit_transform(data['processed_text'])

    features = vectorizer.fit_transform(text_data)

    # Конвертируем признаки (features) в матрицу
    features = features.toarray()

    return vectorizer, features


def make_model_custom_classifier(text_data=data['text'], classifier='SVM'):
    """
        Создание модели и предсказание результатов

        :param text_data: обработанные данные из датасета (после лематизации и пр.)
        :param classifier: указанный пользователем классификатор
        :return: None
    """

    custom_model = None

    # Выбираем классификатор на основе действий пользователя
    if classifier == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        custom_model = KNeighborsClassifier(n_neighbors=1)

    elif classifier == 'Random Forest':
        from sklearn.ensemble import RandomForestClassifier
        custom_model = RandomForestClassifier()

    elif classifier == 'Decision Tree':
        from sklearn.tree import DecisionTreeClassifier
        custom_model = DecisionTreeClassifier()

    elif classifier == 'lda':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        custom_model = LinearDiscriminantAnalysis()

    else:
        from sklearn.svm import SVC
        custom_model = SVC(kernel='linear')

    create_model_time = datetime.datetime.now()
    vectorizer, features = prepare_text_for_model(text_data)

    X_train, X_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.2)

    # Обучаем модель
    print("Модель", classifier, "обучается...")
    custom_model.fit(X_train, y_train)

    # Вычисляем время обучения модели
    evaluation_time = datetime.datetime.now() - create_model_time

    # Предсказываем результат
    y_pred = custom_model.predict(X_test)

    # Вычисляем точность классификации модели
    accuracy = np.mean(y_pred == y_test)

    # Результаты записываем в файл

    cur_path = os.path.dirname(__file__)
    correct_path = os.path.relpath("..\\Reports", cur_path)
    report_path = correct_path + "/model_report.txt"

    with open(report_path, 'a+') as file:
        file.write('********\n')
        file.write(''.join(['Model: ', classifier, '\n']))
        file.write(''.join(['Creation time: ', str(create_model_time), '\n']))
        file.write(''.join(['Evaluation time: ', str(evaluation_time), '\n']))
        file.write(''.join(['Accuracy: ', str(accuracy), '\n']))
        file.write('********\n')


# Особенная, не подходит под другие простые
def make_model_lstm(text_data=data['text']):
    """
        Создание модели на основе LSTM-сетей и предсказание результатов
        P.S. Пока не работает, нужно больше мощностей!

        :param text_data: обработанные данные из датасета (после лематизации и пр.)
        :return: None
    """
    global X_train, X_test, y_train, y_test

    vectorizer, features = prepare_text_for_model(text_data)

    # Divide the dataset into test and training sets.
    X_train, X_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.2)

    # Reshape input data
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print('Размерность X_train:', X_train.shape)
    print('Размерность X_test:', X_test.shape)

    print(u'Собираем модель...')
    print("Len special: ", len(vectorizer.get_feature_names_out()), )
    lstm_model = tf.keras.models.Sequential()
    # lstm_model.add(tf.keras.layers.Embedding(input_dim=500, output_dim=64))
    lstm_model.add(
        tf.keras.layers.LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    lstm_model.add(tf.keras.layers.LSTM(units=64, return_sequences=True))
    lstm_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(lstm_model.summary())

    print(u'Преобразуем категории в матрицу двоичных чисел '
          u'(для использования categorical_crossentropy)')

    num_classes = 2

    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    print(u'Тренируем модель...')
    history = lstm_model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1)
    score = lstm_model.evaluate(X_test, y_test)
    print()
    print(u'Оценка теста: {}'.format(score[0]))
    print(u'Оценка точности модели: {}'.format(score[1]))


def make_model_mine(text_data=data['text']):
    """
        Создание модели на основе MLP (multilayer perceptron) и предсказание результатов

        :param text_data: обработанные данные из датасета (после лематизации и пр.)
        :return: None
    """
    global model
    global X_train, X_test, y_train, y_test
    global vectorizer

    # Время отсчета создания модели
    create_model_time = datetime.datetime.now()

    # Получаем признаки и вектор
    vectorizer, features = prepare_text_for_model(text_data)

    # Определяем архитектуру нейросети
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(len(vectorizer.get_feature_names_out()),)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Компилируем модель
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Приводим столбцы в тип int (для правильной работы в model.fit)
    data['label'] = data['label'].astype(int)

    # Делим выборку на обучающую и тестовую
    X_train, X_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.2)

    # Добавляем кросс-валидационную выборку
    x_train, x_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.2)

    # Обучаем модель
    history = model.fit(
        features, data['label'], epochs=12, batch_size=64, verbose=1,
        validation_data=(x_cv, y_cv)
    )

    # Вычисляем время обучения модели
    evaluation_time = datetime.datetime.now() - create_model_time

    # Вычисляем точность и функцию потерь модели
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Потери на тестах:', loss)
    print('Тестовая точность:', accuracy)
    print('Время обучения:', evaluation_time)

    # Построение графиков
    plt.figure(figsize=(12, 5))

    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Точность на обучении')
    plt.plot(history.history['val_accuracy'], label='Точность при валидации')
    plt.title('График точности')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()

    # График потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Потери на обучении')
    plt.plot(history.history['val_loss'], label='Потери на валидации')
    plt.title('График потерь')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Сохраняем графики в jpg
    save_name = ("mine_model_" + ".jpg")
    plt.savefig(save_name)

    # Получение пути для модели и вектора (т.к. находится все в другой, внешней папке)
    cur_path = os.path.dirname(__file__)
    correct_path = os.path.relpath("..\\ai_model", cur_path)
    model_path = correct_path + "/model.joblib"
    vectorizer_path = correct_path + "/vectorizer.joblib"

    # Сохраняем полученную модель и вектор в отдельный файл (для его повторного использования в других задачах)
    from joblib import dump
    dump(model, model_path, compress=9)
    dump(vectorizer, vectorizer_path, compress=9)


def predict_model(text_feature) -> str:
    """
        Предсказываем результат классификации

        :param text_feature: текстовые признаки
        :return: str: результат классификации (NORN, DOUBT, CONF)
    """
    global model

    # Тестирование
    pred = model.predict(text_feature)

    print("\n")
    str_pred = ""
    result = ""

    for i in pred:
        num_pred = float(i[0])

        if num_pred >= 0.51:
            str_pred = "CONF"
        elif 0.51 >= num_pred >= 0.24:
            str_pred = "DOUBT"
        else:
            str_pred = "NORM"

        result = "".join(["Результат: ", str(i), " Тип: ", str_pred])
        print(result)

    return result


def new_old_vectorizer_process(text_data):
    """
        Получаем features от вектора, с помощью которых будет проходить обучение сети

        :param text_data:
        :return: features - признаки сообщений
    """
    global vectorizer

    # Получаем обработанный текст
    processed_text_data = []
    for i in range(0, len(text_data)):
        processed_text_data.append(preprocess_text(text_data[i]))

    features = vectorizer.transform(processed_text_data)

    # Convert the features to a dense matrix
    features = features.toarray()

    return features


def classify_view_version(my_data) -> str:
    """
        Версия классификации для view

        :param my_data: данные, которые нужно классифицировать
        :return: str: результат классификации
    """
    global model
    global vectorizer

    # Получение пути для модели и вектора (т.к. находится все в другой, внешней папке)
    cur_path = os.path.dirname(__file__)
    correct_path = os.path.relpath("..\\ai_model", cur_path)
    model_path = correct_path + "/model.joblib"
    vectorizer_path = correct_path + "/vectorizer.joblib"

    # Загружаем сохраненные модель и вектор
    from joblib import load
    model = load(model_path)
    vectorizer = load(vectorizer_path)

    # Получаем features
    td = pd.Series(my_data)
    feature_test = new_old_vectorizer_process(td)

    # Предсказываем результат и возвращаем его
    return predict_model(feature_test)


def classify():
    """
        Используется для теста классификации модели (в консоли)

        :return: None
    """

    # Загружаем модель и вектор признаков
    from joblib import load
    model = load('model.joblib')
    vectorizer = load('vectorizer.joblib')

    # Берем кастомные данные (с ввода с консоли пользователем)
    my_data = input("Введите ваше сообщение: ")

    # test_data = "ты не видела куда он тогда пошел? мне очень интересно"
    # processed_text = preprocess_text(test_data)
    td = pd.Series(my_data)

    feature_test = new_old_vectorizer_process(td)

    # Предсказываем результат
    predict_model(feature_test)


# Запускаем, когда нужно пересохранить модель и вектор
if __name__ == "__main__":
    make_model_mine()

    # make_model_custom_classifier(classifier='Random Forest')
    # make_model_custom_classifier(classifier='SVM')
    # make_model_custom_classifier(classifier='lda')
    # make_model_custom_classifier(classifier='KNN')
    # make_model_custom_classifier(classifier='Decision Tree')
    # make_model_lstm()
