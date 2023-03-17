import datetime
import os
import socket
import test

def is_string_path(st):
    cond_file = os.path.isfile(st)
    cond_dir = os.path.isdir(st)

    if cond_file or cond_dir:
        return True
    else:
        return False

def is_file_confidential(path_to_file):

    text = ""
    file = open(path_to_file, "r")

    while True:
        line = file.readline()

        if not line:
            break

        text += line.strip() + " "
    file.close()

    cond1 = test.mail_match(text)
    cond2 = test.phone_number_match(text)
    cond3 = test.passport_data_match(text)
    cond4 = test.credit_card_match(text)

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

        print("This file (" + path_to_file + ") may contain confidential data! TYPE = ", type_conf_data)
        return True
    else:
        print("This file (" + path_to_file + ") has no confidential data")
        return False

def conf_info_detected(data):
    detection_date = datetime.datetime.now()
    detection_date_right_format = str(datetime.datetime.date(datetime.datetime.now()))
    message = " ".join(["Conf file (", data, " ) detected at: ", str(detection_date), " by ", socket.gethostname(), "\n"])
    print(message)

    file_name = detection_date_right_format + "_" + socket.gethostname()
    path = "./Logs/" + file_name + ".txt"
    with open(path, "a") as file:
        file.write(str(message))



if __name__ == "__main__":
    is_file_confidential("Hello! My new neighbor is pretty nice guy. Maybe i should to talk to him soon.")
    is_file_confidential("Hi! I just found out that her number is +79525481672! Take your chance and call her now!!!")
    is_file_confidential("Hi! Plz, i have passed through pure hell and i dont even have money to call you. Plz, send me some money on 4263 2123 3213 6421")