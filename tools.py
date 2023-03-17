import datetime
import socket
import test

def is_text_confidential(text):
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

        print("This text may contain confidential data! TYPE = ", type_conf_data)
    else:
        print("This file has no confidential data")

def conf_info_detected():
    print("Conf file detected at: ", datetime.datetime.now(), " by ", socket.gethostname())


if __name__ == "__main__":
    is_text_confidential("Hello! My new neighbor is pretty nice guy. Maybe i should to talk to him soon.")
    is_text_confidential("Hi! I just found out that her number is +79525481672! Take your chance and call her now!!!")
    is_text_confidential("Hi! Plz, i have passed through pure hell and i dont even have money to call you. Plz, send me some money on 4263 2123 3213 6421")