import os
from service.conf_utils import is_file_or_text_confidential, get_conf_files
from service.file_utils import hash_file


def update_db():
    cur_path = os.path.dirname(__file__)
    db_path = os.path.relpath("..\\view\\fileDatabase", cur_path)

    open(db_path, 'w').close()
    _get_file_hashes()


def _write_file_to_db(elem):
    """
        edit: Сделал private методом
    """

    cur_path = os.path.dirname(__file__)
    db_path = os.path.relpath("..\\view\\fileDatabase", cur_path)

    text_conf = is_file_or_text_confidential(False, elem)
    hash_elem = hash_file(elem)

    with open(db_path, "a") as file:
        file.write(elem + "-----" + hash_elem + "-----" + str(text_conf) + "\n")

    file.close()


def _get_file_hashes(directory='D:\\TEST FOLDER'):
    """
        edit: Сделал private методом
    """

    file_list = list()

    for (dir_path, dir_names, file_names) in os.walk(directory):
        file_list += [os.path.join(dir_path, file) for file in file_names]

    for elem in file_list:
        _write_file_to_db(elem)


def make_conf_file_db():
    cur_path = os.path.dirname(__file__)
    conf_db_path = os.path.relpath("..\\view\\conf_fileDatabase", cur_path)

    with open(conf_db_path, 'w') as conf_file:
        for i in get_conf_files():
            conf_file.write(i + "\n")


def update_conf_db():
    cur_path = os.path.dirname(__file__)
    db_path = os.path.relpath("..\\view\\conf_fileDatabase", cur_path)

    open(db_path, 'w').close()
    make_conf_file_db()

