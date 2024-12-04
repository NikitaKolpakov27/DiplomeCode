# Обучаюшая выборка со спам письмами:
import math
import os.path
from os import listdir
from os.path import isfile
from typing import Any

import service.conf_detect
from service import file_utils

# Письмо требующее проверки
test_letter = "В магазине гора яблок. Купи семь килограмм и шоколадку"

# Создаем новый столбик для подсчета не спам писем
spam_count = 0
not_spam_count = 0


def remove_digits(array=None) -> list[Any]:
    """
        Удаление цифр из массива строк

        :param array: массив, в котором могут быть цифры
        :return: list new_arr: массив без цифр
    """
    new_arr = []

    for i in array:
        try:
            int(i)
            continue
        except ValueError:
            new_arr.append(i)

    return new_arr


def prepare_data(mode):
    """
        Получение данных по выборкам (нормальные и конфиденциальные данные)

        :param str mode: режим данных (определение, из какого набора брать данные: из конфиденциального или нормального)
        :return: list data_array: список, состоящий из слов из файлов в выбранном каталоге
    """

    data_array = []
    cur_path = os.path.dirname(__file__)

    # Опеределение пути к датасету
    if mode == 'normal':
        new_path = os.path.relpath("..\\dataset\\normal", cur_path)
    else:
        new_path = os.path.relpath("..\\dataset\\conf", cur_path)

    # Получаем все файлы из выбранного каталога
    for root, dirs, files in os.walk(new_path):
        for filename in files:
            pdf_file_path = root + "\\" + str(filename)

            # Добавляем в список прочитанные слова из каждого файла
            data_array.append(file_utils.read_pdf_file(pdf_file_path))

    return data_array


def bayes_text_classify(test_text) -> bool:
    """
        Классификация текста на нормальный и конфиденциальный с помощью наивного Байеса

        :param str test_text: текст для классификации
        :return: bool: результат классификации
            * True - конфиденциальный
            * False - нормальный
    """

    # Массивы со спам-словами, со словами без спама и общие
    conf_words = []
    normal_words = []
    total_words = []

    # Получение слов из выборки (конфиденциальной и нормальной)
    conf_array = prepare_data('conf')
    normal_array = prepare_data('normal')

    # Удаление цифр из выборки
    conf_array = remove_digits(conf_array)
    normal_array = remove_digits(normal_array)

    # Формирование массива спам-слов
    for i in conf_array:
        conf_letter = service.conf_detect.preprocessing(i)

        for conf_word in conf_letter:
            conf_words.append(conf_word)
            total_words.append(conf_word)

    # Формирование массива нормальных слов
    for i in normal_array:
        normal_letter = service.conf_detect.preprocessing(i)

        for normal_word in normal_letter:
            normal_words.append(normal_word)
            total_words.append(normal_word)

    # Удаление дубликатов из обучающей выборки
    for i in total_words:
        if total_words.count(i) > 1:
            total_words.remove(i)

    # Цикл по тест. письму
    normalized_test_letter = service.conf_detect.preprocessing(test_text)
    total_probability_CONF = math.log(len(conf_array) / (len(conf_array) + len(normal_array)))
    total_probability_NORMAL = math.log(len(normal_array) / (len(conf_array) + len(normal_array)))

    conf_multiplier = total_probability_CONF
    normal_multiplier = total_probability_NORMAL

    for i in normalized_test_letter:

        if i in total_words:
            conf_counter = conf_words.count(i)
            normal_counter = normal_words.count(i)

            conf_result = math.log((conf_counter + normal_counter) / (len(total_words) + len(conf_words)))
            normal_result = math.log((conf_counter + normal_counter) / (len(total_words) + len(normal_words)))

            conf_multiplier += conf_result
            normal_multiplier += normal_result

        else:
            conf_result = math.log(1 / (len(total_words) + len(conf_words)))
            normal_result = math.log(1 / (len(total_words) + len(normal_words)))

            conf_multiplier += conf_result
            normal_multiplier += normal_result

    print("CONF probability: ", conf_multiplier)
    print("NORMAL probability: ", normal_multiplier)

    return conf_multiplier > normal_multiplier


if __name__ == "__main__":
    r = bayes_text_classify(test_text="Привет, как дела? Что делаешь? Слугай, как насчет всместе сходить в кино на"
                                      "следующей неделе? Что думаешь?")
    print(r)
    # n_arr = prepare_data(mode='normal')
    # print(n_arr)
