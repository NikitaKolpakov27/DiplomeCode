import datetime
import os
import socket
import service.conf_detect as conf_detect
from service.file_utils import get_file_type, read_docx_file, read_pdf_file, read_txt_file
from docx.opc.exceptions import PackageNotFoundError
import service.reg_exp_utils as reg_exp_utils


# Получает хэши конфиденциальных файлов
def get_conf_hashes():
    raw_array = []
    conf_hashes_array = []

    with open("./fileDatabase", "r") as file:
        lines = file.readlines()

        for line in lines:
            raw_array.append(line.split("-----"))

    for i in range(0, len(raw_array)):

        if raw_array[i][2] == "True\n":
            conf_hashes_array.append(raw_array[i][1])

    return conf_hashes_array


def is_file_or_text_confidential(is_text, path_to_file):
    text = ""

    # Если файл представляет собой путь к файлу
    if not is_text:

        try:

            file_type = get_file_type(path_to_file)

            if file_type == ".pdf":
                text = read_pdf_file(path_to_file)

            elif file_type == ".docx":
                text = read_docx_file(path_to_file)

            else:
                text = read_txt_file(path_to_file)

        except PackageNotFoundError:
            print("Couldn't find *.doc or *.docx file. Maybe, it was deleted!")
            return True
        except FileNotFoundError:
            print("Couldn't find file. Maybe, it was deleted!")
            return True

    else:
        text = path_to_file

    # Проверка текста на регулярки
    cond1 = reg_exp_utils.mail_match(text)
    cond2 = reg_exp_utils.phone_number_match(text)
    cond3 = reg_exp_utils.passport_data_match(text)
    cond4 = reg_exp_utils.credit_card_match(text)
    cond5 = reg_exp_utils.ipv4_match(text)
    cond6 = reg_exp_utils.password_match(text)
    cond7 = reg_exp_utils.ipv6_match(text)
    cond8 = reg_exp_utils.mac_address_match(text)

    if cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or cond7 or cond8:
        return True
    else:

        # Проверка текста на ключевые слова
        precetnage_conf = conf_detect.check_conf_info(text)

        if precetnage_conf > 10:
            return True

        return False


# Отправка отчёта после операций с конфиденциальными файлами
# ЗАПИСАТЬ СООБЩЕНИЕ В МЕССЕДЖ БОКС!!!
def conf_info_detected(data, action):
    detection_date = datetime.datetime.now()
    detection_date_right_format = str(datetime.datetime.date(datetime.datetime.now()))
    message = " ".join(
        ["Actions (" + action + ") with conf file (", data, " ) were detected at: ", str(detection_date), " by ",
         socket.gethostname(), "\n"])
    print(message)

    file_name = detection_date_right_format + "_" + str(detection_date.hour) + "_" \
                + str(detection_date.minute) + "_" + str(detection_date.second) + "_" \
                + socket.gethostname()

    cur_path = os.path.dirname(__file__)
    correct_path = os.path.relpath('..\\Reports', cur_path)
    correct_path = correct_path + "/" + file_name + ".txt"

    with open(correct_path, "a") as file:
        file.write(str(message))

    return message


if __name__ == "__main__":
    try:
        f = read_docx_file("D:\\TEST FOLDER\\Лабораторная_11.docx")
        print(f)
    except PackageNotFoundError:
        print("S docx")
    except FileNotFoundError:
        print("S file")
