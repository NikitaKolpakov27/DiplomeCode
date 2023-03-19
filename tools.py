import datetime
import hashlib
import os
import socket
import reg_exp_conf


def update_db():
    open('./fileDatabase', 'w').close()
    get_file_hashes()


def write_file_to_db(elem):
    text_conf = is_file_or_text_confidential(False, elem)
    hash_elem = hash_file(elem)

    with open("./fileDatabase", "a") as file:
        file.write(elem + "-----" + hash_elem + "-----" + str(text_conf) + "\n")

    file.close()


def get_file_hashes(directory='D:\\На новый ноут\\Учёба\\TEST FOLDER'):
    file_list = list()

    for (dir_path, dir_names, file_names) in os.walk(directory):
        file_list += [os.path.join(dir_path, file) for file in file_names]

    for elem in file_list:
        write_file_to_db(elem)


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


def is_file_or_text_confidential(is_text, path_to_file):
    text = ""

    if not is_text:

        try:

            file = open(path_to_file, "r")

            while True:
                line = file.readline()

                if not line:
                    break

                text += line.strip() + " "

        except FileNotFoundError:
            print("Couldn't find file. Maybe, it was deleted!")
            return True

        file.close()
    else:
        text = path_to_file

    cond1 = reg_exp_conf.mail_match(text)
    cond2 = reg_exp_conf.phone_number_match(text)
    cond3 = reg_exp_conf.passport_data_match(text)
    cond4 = reg_exp_conf.credit_card_match(text)

    if cond1 or cond2 or cond3 or cond4:
        type_conf_data = ""

        if cond1:
            type_conf_data += "Mail address, "
        if cond2:
            type_conf_data += "Phone number, "
        if cond3:
            type_conf_data += "Passport data, "
        if cond4:
            type_conf_data += "Credit card info "

        if not is_text:
            pass
            # print("This file (" + path_to_file + ") may contain confidential data! TYPE = ", type_conf_data)
        return True
    else:
        if not is_text:
            pass
            # print("This file (" + path_to_file + ") has no confidential data")
        return False


def conf_info_detected(data, action):
    detection_date = datetime.datetime.now()
    detection_date_right_format = str(datetime.datetime.date(datetime.datetime.now()))
    message = " ".join(
        ["Actions (" + action + ") with conf file (", data, " ) were detected at: ", str(detection_date), " by ",
         socket.gethostname(), "\n"])
    print(message)

    file_name = detection_date_right_format + "_" + str(detection_date.hour) + "_" + str(detection_date.minute) \
                + "_" + str(detection_date.second) + "_" + socket.gethostname()
    path = "./Reports/" + file_name + ".txt"
    with open(path, "a") as file:
        file.write(str(message))


if __name__ == "__main__":
    is_file_or_text_confidential(True, "Hello! My new neighbor is pretty nice guy. Maybe i should to talk to him soon.")
    is_file_or_text_confidential(True,
                                 "Hi! I just found out that her number is +79525481672! Take your chance and call her now!!!")
