import os
from conf_utils import is_file_or_text_confidential
from file_utils import hash_file


def update_db():
    open('./fileDatabase', 'w').close()
    get_file_hashes()


def write_file_to_db(elem):
    text_conf = is_file_or_text_confidential(False, elem)
    hash_elem = hash_file(elem)

    with open("./fileDatabase", "a") as file:
        file.write(elem + "-----" + hash_elem + "-----" + str(text_conf) + "\n")

    file.close()


def get_file_hashes(directory='D:\\TEST FOLDER'):
    file_list = list()

    for (dir_path, dir_names, file_names) in os.walk(directory):
        file_list += [os.path.join(dir_path, file) for file in file_names]

    for elem in file_list:
        write_file_to_db(elem)
