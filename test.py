import re

#Работает!!!
def phone_number_match(msg):
    # msg = "Hello. Sorry for no answer, i was very busy. But now, u can call my by the number +74734521565"
    matches = re.findall("[+]?[7-8]{1}[0-9]{10}", msg)

    if len(matches) > 0:
        return True

    return False

#Работает, но не очень правильно
def credit_card_match(msg):
    # msg = "Hello, my series and number of password are 4276 2131 3215 6431"
    matches = re.findall("[1-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?", msg)

    if len(matches) > 0:
        return True

    return False

#Работает!!!
def mail_match(msg):
    # msg = "Hello, my email is sandore24@bk.co"
    matches = re.findall("[a-zA-Z0-9._]+@[a-z]+\.[a-z]{2,4}", msg)

    if len(matches) > 0:
        return True

    return False

#Работает!!!
def passport_data_match(msg):
    # msg = "Hello, my series and number of password are 10 02 824113"
    matches = re.findall("\d{2}[^0-9]*\d{2}[ -,_/]+\d{6}", msg)

    if len(matches) > 0:
        return True

    return False


if __name__ == "__main__":
    passport_data_match("89 52 548167")
