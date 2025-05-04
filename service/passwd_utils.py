import os
import random
import bcrypt


def create_passwd(password):
    """
        Создание пароля (его хеша) для входа в DLP-систему из случайных букв и цифр

        :return: str hashed_password: строку-пароль в виде хеша
    """
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password_bytes, salt)

    cur_path = os.path.dirname(__file__)
    correct_path = os.path.relpath("..\\view", cur_path)
    passwd_file = correct_path + "\\passwd.txt"

    with open(passwd_file, "wb") as file:
        file.write(hashed_password)

    return hashed_password


def get_passwd():
    """
        Получение пароля (хеша) из файла

        :return: str passwd: пароль, записанный в файле
    """
    with open("./passwd.txt", "rb+") as f:
        passwd = f.read()

    return passwd


def check_passwd(entered_passwd) -> bool:
    """
        Проверка введенного пароля. Сравнение его с имеющейся записью в файле.

        :param str entered_passwd: введенный пользователем пароль
        :return: bool: результат сходства между введенным и имеющимся паролем
            * True - пароли совпадают
            * False - не совпадают
    """
    return bcrypt.checkpw(entered_passwd.encode('utf-8'), get_passwd())
    # right_passwd = get_passwd()
    #
    # return entered_passwd == right_passwd
