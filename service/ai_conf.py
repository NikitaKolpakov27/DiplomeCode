import datetime
import os
import string
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss
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
data = pd.read_csv(conf_dataset, encoding='utf-8', on_bad_lines='error')


def print_dataset(dataset):
    print("Текст:")
    for i in range(0, len(dataset)):
        print(i + 1, " => ", dataset[i])
    print("================================")


def from_pandas_to_dict(data):
    """
        Из массива типа DataFrame переводим в словарь (типа: "сообщение" - "ключ")

        :param data: данные в виде DataFrame
        :return: датасет в виде словаря
    """
    texts = data['text']
    labels = data['label']

    texts = texts.astype(str)
    labels = labels.astype(str)

    new_dict = dict(zip(texts, labels))
    return new_dict


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


def prepare_text_for_model(train_texts, test_texts=None):
    """
        Векторизация текста с правильным разделением на train/test

        :param train_texts: тексты для обучения
        :param test_texts: тексты для тестирования (опционально)
        :return: кортеж (vectorizer, train_features, test_features)
    """
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 4),
        min_df=3,
        max_df=0.93,
        stop_words='english',
        sublinear_tf=True,
        binary=True
    )

    # Обучение векторизатора только на тренировочных данных
    train_features = vectorizer.fit_transform(train_texts).toarray()

    # Преобразование тестовых данных (если переданы)
    test_features = vectorizer.transform(test_texts).toarray() if test_texts is not None else None

    return vectorizer, train_features, test_features

def make_model_custom_classifier(classifier='SVM'):
    """
        Создание модели и предсказание результатов

        :param text_data: обработанные данные из датасета (после лематизации и пр.)
        :param classifier: указанный пользователем классификатор
        :return: None
    """

    data['processed_text'] = data['text'].apply(preprocess_text)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        data['processed_text'],
        data['label'].astype(int),
        test_size=0.2,
        random_state=42
    )

    # 2. Векторизация с правильным разделением
    vectorizer, X_train, X_test = prepare_text_for_model(X_train_raw, X_test_raw)

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

    elif classifier == 'Logistic Regression':
        from sklearn.linear_model import LogisticRegression
        custom_model = LogisticRegression()

    elif classifier == 'Bayes':
        from sklearn.naive_bayes import GaussianNB
        custom_model = GaussianNB()

    else:
        from sklearn.svm import SVC
        custom_model = SVC(kernel='linear')

    create_model_time = datetime.datetime.now()

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
        file.write(''.join(['Precision: ', str(precision_score(y_pred, y_test)), '\n']))
        file.write(''.join(['Recall: ', str(recall_score(y_pred, y_test)), '\n']))
        file.write(''.join(['F1-score: ', str(f1_score(y_pred, y_test)), '\n']))
        file.write(''.join(['ROC-AUC: ', str(roc_auc_score(y_pred, y_test)), '\n']))
        file.write(''.join(['Log Loss: ', str(log_loss(y_pred, y_test)), '\n']))
        file.write('********\n')
    file.close()


def make_model_mine():
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

    # Предобработка текста
    data['processed_text'] = data['text'].apply(preprocess_text)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        data['processed_text'],
        data['label'].astype(int),
        test_size=0.2,
        random_state=42
    )

    # 2. Векторизация с правильным разделением
    vectorizer, X_train, X_test = prepare_text_for_model(X_train_raw, X_test_raw)

    print("Len: ", len(X_train[0]))

    # Определяем архитектуру нейросети
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(128, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Компилируем модель
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Приводим столбцы в тип int (для правильной работы в model.fit)
    data['label'] = data['label'].astype(int)

    # Добавляем кросс-валидационную выборку
    X_train, x_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.2)

    # Обучаем модель
    history = model.fit(
        X_train, y_train, epochs=11, batch_size=128, verbose=1,
        validation_data=(x_cv, y_cv),
    )

    # Вычисляем время обучения модели
    evaluation_time = datetime.datetime.now() - create_model_time

    # Вычисляем точность и функцию потерь модели
    loss, accuracy = model.evaluate(X_test, y_test)

    # Получаем предсказанные результаты модели (для метрик)
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Выводим метрики в консоль
    print('Потери на тестах:', loss)
    print('Тестовая точность:', accuracy)
    print('Время обучения:', evaluation_time)
    print('Precision: ', str(precision_score(y_pred, y_test)))
    print('Recall: ', str(recall_score(y_pred, y_test)))
    print('F1-Score: ', str(f1_score(y_pred, y_test)))
    print('ROC-AUC: ', str(roc_auc_score(y_pred, y_test)))

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

    # Точность с шумом
    X_test_noisy = X_test + np.random.normal(0, 0.05, X_test.shape)  # Добавление 5% шума
    loss, accuracy = model.evaluate(X_test_noisy, y_test)
    print("Точность с шумом: ", accuracy)

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

        result = (i, str_pred)
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

    # Логирование (потом убрать)
    print("Слова (обработанные): ", processed_text_data)
    print("Слова: ", ''.join(processed_text_data).split(" "))
    print("Хар-ка слов: ", features[0], "Features len: ", len(features[0]))

    print("Size of tokens: ", len(processed_text_data))
    print("Size of features: ", len(features))

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
    global model
    global vectorizer

    # Получение пути для модели и вектора (т.к. находится все в другой, внешней папке)
    cur_path = os.path.dirname(__file__)
    correct_path = os.path.relpath("..\\ai_model", cur_path)
    model_path = correct_path + "/model.joblib"
    vectorizer_path = correct_path + "/vectorizer.joblib"

    # Загружаем модель и вектор признаков
    from joblib import load
    model = load(model_path)
    vectorizer = load(vectorizer_path)

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
    # make_model_custom_classifier(classifier='Logistic Regression')
    # make_model_custom_classifier(classifier='KNN')
    # make_model_custom_classifier(classifier='Decision Tree')
    # make_model_custom_classifier(classifier='Bayes')
    # make_model_lstm()

