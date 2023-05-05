# Обучаюшая выборка со спам письмами:
import os.path
from os import listdir
from os.path import isfile

import service.conf_detect
from service import file_utils

conf_array = (
    'Кстати, мне нужно чтоб ты зашел на сайт через меня. Щас скину свой логин и пароль....',
    'Перечисли мне 5000 рублей на карту. Номер карты скину позже',
    'Это сообщение содержит сведения конфиденциального характера'
)

# Обучающая выборка с "нормальными" письмами:
normal_array = (
    'Завтра состоится собрание',
    'Купи килограмм яблок и шоколадку',
    ''
)

# Письмо требующее проверки
test_letter = "В магазине гора яблок. Купи семь килограмм и шоколадку"

# Создаем новый столбик для подсчета не спам писем
spam_count = 0
not_spam_count = 0

def prepare_normal_data():
    norm_array = []

    cur_path = os.path.dirname(__file__)
    new_path = os.path.relpath("..\\dataset\\normal", cur_path)

    for root, dirs, files in os.walk(new_path):
        for filename in files:
            pdf_file_path = root + "\\" + str(filename)

            norm_array.append(file_utils.read_pdf_file(pdf_file_path))

    return norm_array


def bayes_text_classify(test_letter):

    # Массивы со спам-словами, со словами без спама и общие
    conf_words = []
    normal_words = []
    total_words = []

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
    normalized_test_letter = service.conf_detect.preprocessing(test_letter)
    total_probability_CONF = len(conf_array) / (len(conf_array) + len(normal_array))
    total_probability_NORMAL = len(normal_array) / (len(conf_array) + len(normal_array))

    conf_multiplier = total_probability_CONF
    normal_multiplier = total_probability_NORMAL

    for i in normalized_test_letter:

        if i in total_words:
            conf_counter = total_words.count(i)
            normal_counter = total_words.count(i)

            conf_result = (conf_counter + normal_counter) / (len(total_words) + len(conf_words))
            normal_result = (conf_counter + normal_counter) / (len(total_words) + len(normal_words))

            conf_multiplier *= conf_result
            normal_multiplier *= normal_result

        else:
            conf_result = 1 / (len(total_words) + len(conf_words))
            normal_result = 1 / (len(total_words) + len(normal_words))

            conf_multiplier *= conf_result
            normal_multiplier *= normal_result

    print("CONF probability: ", conf_multiplier)
    print("NORMAL probability: ", normal_multiplier)

    return conf_multiplier > normal_multiplier


if __name__ == "__main__":
    # r = bayes_text_classify(test_letter=test_letter)
    # print(r)
    n_arr = prepare_normal_data()
    print(n_arr)