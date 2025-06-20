import json
import os
from service import ai_conf
from service.file_utils import get_file_type, read_docx_file, read_pdf_file, read_txt_file, write_log
from docx.opc.exceptions import PackageNotFoundError
import service.reg_exp_utils as reg_exp_utils

# Получает хэши конфиденциальных файлов
def get_conf_hashes():
    conf_hashes_array = []

    cur_path = os.path.dirname(__file__)
    db_path = os.path.relpath("..\\view\\fileDB.json", cur_path)

    with open(db_path, "r", encoding="utf-8") as file:
        fileDB = json.load(file)
        total_files = len(fileDB["files"])

    for i in range(0, total_files):

        if fileDB["files"][i]["status"]:
            conf_hashes_array.append(fileDB["files"][i]["file_hash"])

    return conf_hashes_array


# Получает имена конфиденциальных файлов (из основной БД)
def get_conf_files():
    raw_array = []
    conf_files_array = []

    cur_path = os.path.dirname(__file__)
    db_path = os.path.relpath("..\\view\\fileDB.json", cur_path)

    with open(db_path, "r", encoding="utf-8") as file:
        fileDB = json.load(file)
        total_files = len(fileDB["files"])

    for i in range(0, total_files):

        if fileDB["files"][i]["status"]:
            conf_files_array.append(fileDB["files"][i]["file_path"])

    return conf_files_array


def is_file_was_in_db(suspect_file):
    suspect_file = suspect_file.replace("/", "\\")
    suspect_file = suspect_file + "\n"

    cur_path = os.path.dirname(__file__)
    db_path = os.path.relpath("..\\view\\conf_fileDatabase.txt", cur_path)

    with open(db_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

        for line in lines:
            if suspect_file == line.lower():
                return True

        return False


def is_file_or_text_confidential(is_text, path_to_file) -> bool:
    """
        Проверка текста на наличие в нем признаков конфиденциальной информации

        :param bool is_text: Является ли переданный файл файлом (или просто текстом)?
        :param str path_to_file: Путь к файлу

        :return: bool: есть ли признаки конфиденциальной информации в файле или нет?
    """
    text = ""

    # Если файл представляет собой путь к файлу
    if not is_text:

        # Проверяет, есть ли файл в базе конф. файлов. Есть - значит, определенно содержит признаки
        if is_file_was_in_db(path_to_file):
            return True

        try:

            file_type = get_file_type(path_to_file)

            if file_type == ".pdf":
                text = read_pdf_file(path_to_file)

            elif file_type == ".docx":
                text = read_docx_file(path_to_file)

            elif file_type == ".txt":
                text = read_txt_file(path_to_file)

            # Случай с неподдерживаемыми типами для рассмотрения
            else:
                text = "default"

        except PackageNotFoundError:
            print("Couldn't find *.doc or *.docx file. Maybe, it was deleted!")
            return False
        except FileNotFoundError:
            print("Couldn't find file. Maybe, it was deleted!")
            return False

    else:
        text = path_to_file

    # Проверка текста на регулярки
    cond1 = reg_exp_utils.mail_match(text)
    cond2 = reg_exp_utils.phone_number_match(text)
    cond3 = reg_exp_utils.passport_data_match(text)
    cond4 = reg_exp_utils.credit_card_match(text)
    cond5 = reg_exp_utils.ipv4_match(text)
    cond6 = reg_exp_utils.ipv6_match(text)
    cond7 = reg_exp_utils.mac_address_match(text)
    cond8 = reg_exp_utils.inn_match(text)
    cond9 = reg_exp_utils.snils_match(text)

    if cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or cond7 or cond8 or cond9:
        return True
    else:

        # Проверка текста с помощью ИИ
        # precetnage_conf, status = conf_detect.check_conf_info(text)
        precetnage_conf, status = ai_conf.classify_view_version(text)

        if status == 'NORM':
            return False

        return True


def conf_info_detected(data, action) -> str:
    """
        Отправка отчёта после операций с конфиденциальными файлами

        :param data: данные, в которых был зафиксирован инцидент (имя файла)
        :param action: действие, при котором был зафиксирован инцидент (удаление, перемещение и т.д.)

        :return: str message: само сообщение в строковом виде
    """

    # Формирование сообщения
    message = " ".join([action + " conf file     ", data])
    print(message)

    # Запись отчета о проишествии в лог-файл
    write_log(message)

    return message


if __name__ == "__main__":
    get_conf_files()

