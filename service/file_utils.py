import os
import hashlib
from docx import Document
from PyPDF2 import PdfReader


def read_pdf_file(pdf_path) -> str:
    """
        Чтение PDF-файла и получение его содержимого в виде текста

        :param str pdf_path: путь к PDF-файлу
        :return: str text: строка с текстовым содержимым данного файла
    """

    # Читаем файл и получаем кол-во его страниц
    pdf_file = PdfReader(pdf_path)
    number_of_pages = len(pdf_file.pages)

    text = ""

    # В цикле по страницам файла получаем содержимое и записываем все в строку
    for i in range(0, number_of_pages - 1):
        page = pdf_file.pages[i]
        page_text = page.extract_text()
        text += page_text

    return text


def read_docx_file(docx_path) -> str:
    """
        Чтение docx-файла и получение его содержимого в виде текста

        :param str docx_path: путь к docx-файлу
        :return: str text: строка с текстовым содержимым данного файла
    """
    doc = Document(docx_path)

    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)

    return '\n'.join(text)


def read_txt_file(txt_path) -> str:
    """
        Чтение txt-файла и получение его содержимого в виде текста

        :param str txt_path: путь к txt-файлу
        :return: str text: строка с текстовым содержимым данного файла
    """

    text = ""
    file = open(txt_path, "r")

    while True:
        line = file.readline()

        if not line:
            break

        text += line.strip() + " "
    file.close()

    return text


def get_file_type(file_path) -> str:
    """
        Получаем тип файла (PDF, docx, txt) по его расширению

        :param str file_path: путь к файлу
        :return: str file_extension: расширение файла
    """
    file_extension = os.path.splitext(file_path)[1]

    return file_extension


def hash_file(filename) -> str:
    """
        Процесс получения хэша для файла

        :param str filename: путь к файлу
        :return: str: сформированный хэш файла
    """

    if os.path.isfile(filename) is False:
        raise Exception("File not found for hash operation")

    h_sha256 = hashlib.sha256()

    with open(filename, 'rb') as file:
        chunk = 0
        while chunk != b'':
            chunk = file.read(1024)
            h_sha256.update(chunk)

    return h_sha256.hexdigest()


def is_string_path(path) -> bool:
    """
        Определение, является ли данная строка - путем к файлу или директории

        :param path: проверяемая строка
        :return: bool:
            * True - строка = путь к файлу или директории
            * False - строка = что-то другое
    """
    cond_file = os.path.isfile(path)
    cond_dir = os.path.isdir(path)

    if cond_file or cond_dir:
        return True
    else:
        return False
