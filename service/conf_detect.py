from typing import List, Any

import matplotlib.pyplot as plt
import string
import nltk
import pymorphy3
from nltk import word_tokenize
from wordcloud import WordCloud
from nltk.corpus import stopwords

from service import ai_conf

conf_words_array = ["тайна", "ограниченный", "доступ", "запрещено", "конфиденциально", "конфиденциальный",
                    "информация", "логин", "пароль", "токен", "карта", "кредитка", "паспорт", "секретно",
                    "секрет", "данные", "личное", "гостайна", "конфиденциальность"]

conf_phrases = ["персональные данные", "ограниченный доступ", "государственная тайна", "служебный пользование",
                "совершенно секретно", "конфиденциальная информация", "кредитная карта", "номер телефона",
                "служебная тайна", "врачебная тайна", "тайна судопроизводства", "семейная тайна",
                "коммерческая тайна"]

# Добавил из-за ошибок при компиляции
nltk.download("punkt_tab")
nltk.download("stopwords")

conf_words = ""
for word in conf_words_array:
    conf_words += word + " "


def show_conf_wordcloud(text_array):
    """
        Показывает облако наиболее встречаемых слов для анализируемого текста

        :param text_array: Массив слов
        :return: None
    """

    # Преобразование массива слов в одну строку
    text_array_str = ""
    for i in text_array:
        text_array_str += i + " "

    conf_wordcloud = WordCloud(width=500, height=300).generate(text_array_str)

    plt.figure(figsize=(10, 8), facecolor='w')
    plt.imshow(conf_wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


def preprocessing(text) -> list[Any]:
    """
        Подготовка текста к классификации

        :param text: Проверяемый текст
        :return: массив слов без всякого лишнего (спецсимволы, предлоги и пр.)
    """

    # Удаление спец символов из текста
    text = text.lower()
    spec_chars = string.punctuation + '\xa0«»\t—…'
    text = "".join([ch for ch in text if ch not in spec_chars])

    # Токенизация
    text_tokens = word_tokenize(text)
    text = nltk.Text(text_tokens)
    text_words = text[:len(text)]

    # Удаление стоп-слов
    rus_stopwords = stopwords.words("russian")
    eng_stopwords = stopwords.words("english")

    # Удаление сначала стоп-слов из русского языка
    filtered_tokens_ru = []
    for token in text_words:
        if token not in rus_stopwords:
            filtered_tokens_ru.append(token)

    # Удаление потом стоп-слов и из английского языка
    filtered_tokens = []
    for token in filtered_tokens_ru:
        if token not in eng_stopwords:
            filtered_tokens.append(token)

    # Лемматизация
    final_array = []
    morph = pymorphy3.MorphAnalyzer()

    for word in filtered_tokens:
        p = morph.parse(word)[0]
        final_array.append(p.normal_form)

    return final_array


def find_phrases(array) -> int:
    """
        Поиск конфиденциальных фраз в тексте

        :param array: Массив текста
        :return: int len(phrases): количество обнаруженных конфиденциальных фраз в тексте
    """
    phrases = []

    for i in range(0, len(array)):

        for phrase in conf_phrases:
            phrase_1, phrase_2 = phrase.split(" ")

            if array[i] == phrase_1:
                j = i + 1
                if array[j] == phrase_2:
                    phrases.append(array[i] + " " + array[j])

    return len(phrases)


def check_conf_info(text) -> str:
    """
        Проверяем текст на наличие в нем конфиденциальной информации

        :param text: Проверяемый текст
        :return: float percentage: процент конфиденциальной информации в тексте
                (согласно совпадениям по ключевым словам)
    """
    final_text = preprocessing(text)
    return ai_conf.classify_view_version(final_text)
    # count = 0
    #
    # # Поиск по ключевым словам
    # for i in final_text:
    #     if i in conf_words_array:
    #         count += 1
    #
    # # Поиск по ключевым фразам
    # also_count = find_phrases(final_text)
    # count += also_count
    #
    # # Вычисления процента конфиденциальной информации в тексте
    # length_text = len(final_text) + 1
    # percentage = (count / length_text) * 100
    #
    # return percentage


if __name__ == "__main__":
    # text = file_utils.read_pdf_file("C:/Users/MateBook/Downloads/MEGAShPORA.pdf")
    # check_conf_info(text)

    # txt = "Этот документ только для служебного пользования!"
    # find_phrases(preprocessing(txt))
    arr = "Привет, Аня! Не спишь там в такую погоду?) Напоминаю, что у нас встреча в 10:00 в понедельник. Не забудь принести отчет!".split(" ")
    show_conf_wordcloud(arr)
