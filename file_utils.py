import os
import hashlib
from docx import Document
from PyPDF2 import PdfReader

def read_pdf_file(pdf_path):
    pdf_file = PdfReader(pdf_path)
    number_of_pages = len(pdf_file.pages)

    text = ""
    for i in range(0, number_of_pages - 1):
        page = pdf_file.pages[i]
        page_text = page.extract_text()
        text += page_text

    return text


def read_docx_file(docx_path):
    doc = Document(docx_path)

    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)

    return '\n'.join(text)


def read_txt_file(txt_path):
    text = ""
    file = open(txt_path, "r")

    while True:
        line = file.readline()

        if not line:
            break

        text += line.strip() + " "
    file.close()

    return text


def get_file_type(file_path):
    file_extension = os.path.splitext(file_path)[1]
    return file_extension


def hash_file(filename):
    if os.path.isfile(filename) is False:
        raise Exception("File not found for hash operation")

    h_sha256 = hashlib.sha256()

    with open(filename, 'rb') as file:
        chunk = 0
        while chunk != b'':
            chunk = file.read(1024)
            h_sha256.update(chunk)

    return h_sha256.hexdigest()


def is_string_path(st):
    cond_file = os.path.isfile(st)
    cond_dir = os.path.isdir(st)

    if cond_file or cond_dir:
        return True
    else:
        return False
