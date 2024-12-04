import random


def create_passwd():
    """
        Создание пароля для входа в DLP-систему из случайных букв и цифр

        :return: str password: строку-пароль длиной в 10 символов
    """
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
    password = ''
    for i in range(10):
        password += random.choice(chars)

    return password


def update_passwd():
    """
        Обновление пароля для входа в DLP-систему
        Нужно для надежности защиты (ибо пароль генерируется не слишком сложный => его нужно постоянно менять)

        :return: None
    """
    open("./passwd", 'w').close()
    with open("./passwd", "w") as f:
        f.write(create_passwd())


def get_passwd():
    """
        Получение пароля из файла

        :return: str passwd: пароль, записанный в файле
    """
    with open("./passwd") as f:
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
    right_passwd = get_passwd()

    return entered_passwd == right_passwd
