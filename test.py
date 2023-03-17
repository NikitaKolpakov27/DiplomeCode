import re


def phone_number_match(msg):
    matches = re.findall("[+]?[7-8]{1}[0-9]{10}", msg)

    if len(matches) > 0:
        return True

    return False

def credit_card_match(msg):
    matches = re.findall("[1-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?", msg)

    if len(matches) > 0:
        return True

    return False

def mail_match(msg):
    matches = re.findall("[a-zA-Z0-9._]+@[a-z]+\.[a-z]{2,4}", msg)

    if len(matches) > 0:
        return True

    return False

def passport_data_match(msg):
    matches = re.findall("\d{2}[^0-9]*\d{2}[ -,_/]+\d{6}", msg)

    if len(matches) > 0:
        return True

    return False


if __name__ == "__main__":
    # passport_data_match("89 52 548167")
    # get_file_hashes("D:\\На новый ноут\\Учёба\\TEST FOLDER")

    f = open('D:\\На новый ноут\\Учёба\\TEST FOLDER\\bithc.txt')
    if f.closed:
        print('file is closed')
    else:
        print("Opened")
