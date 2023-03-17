import base64
import random
import string
from os import path

import pyperclip
from PIL import Image, ImageGrab
import hashlib
import tkinter as tk

#Функция, формирующая массив из хешей конфиденциальных файлов
import tools

"""
    Функция, возвращающая хэши конфиденциальных файлов
"""
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


"""
    Функция, которая хэширует файлы
"""
def hash_file(filename):
    if path.isfile(filename) is False:
        raise Exception("File not found for hash operation")

    h_sha256 = hashlib.sha256()

    with open(filename, 'rb') as file:
        chunk = 0
        while chunk != b'':
            chunk = file.read(1024)
            h_sha256.update(chunk)

    return h_sha256.hexdigest()

def saveLetter(data, path):
    with open(path, "w") as file:
        file.write(data)


def from_pic_to_str(pic_path, bs4_path):
    with open(pic_path, "rb") as image2string:
        converted_string = base64.b64encode(image2string.read())

    with open(bs4_path, "wb") as file:
        file.write(converted_string)


# def from_str_to_pic():
#     file = open('encode.bin', 'rb')
#     byte = file.read()
#     file.close()
#
#     decodeit = open('hello_level.jpeg', 'wb')
#     decodeit.write(base64.b64decode((byte)))
#     decodeit.close()

def get_file_from_clipboard():
    last_data = None

    conf_hashes = get_conf_hashes()

    while True:

        root = tk.Tk()
        root.withdraw()
        data = root.clipboard_get()
        data_type = tools.is_string_path(data)

        if data != last_data and len(data) != 0:

            if data_type:

                print(data)
                print(hash_file(data))

                if hash_file(data) in conf_hashes:
                    print("WARNING!")
                    tools.conf_info_detected(data)
                    root.clipboard_clear()
                    break
            else:
                print(data)

        last_data = data

def get_text_data_from_clipboard():
    last_data = None

    while True:

        data = pyperclip.paste()

        if data != last_data and len(data) != 0:
            print(data)
            print("Last data: ", last_data)

        last_data = data


if __name__ == "__main__":
    pass
    # get_text_data_from_clipboard()
    # md5txt = hash_file("D:\\На новый ноут\\Учёба\\TEST FOLDER\\bithc.txt")
    # print("needed text: ", md5txt)
    #
    get_file_from_clipboard()
    # get_pic_from_clipboard()


