import re


def phone_number_match(msg):
    matches = re.findall("[+]?[7-8][0-9]{10}", msg)
    # print("Matches: ", matches)

    if len(matches) > 0:
        return True

    return False

def credit_card_match(msg):
    matches = re.findall("[1-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?", msg)
    # print("Matches: ", matches)

    if len(matches) > 0:
        return True

    return False

def mail_match(msg):
    matches = re.findall("[a-zA-Z0-9._]+@[a-z]+\.[a-z]{2,4}", msg)
    # print("Matches: ", matches)

    if len(matches) > 0:
        return True

    return False

def passport_data_match(msg):
    matches = re.findall("\d{2}[^0-9]*\d{2}[ -,_/]+\d{6}", msg)
    # print("Matches: ", matches)

    if len(matches) > 0:
        return True

    return False

def ipv4_match(msg):
    matches = re.findall("((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)", msg)

    if len(matches) > 0:
        return True

    return False

def ipv6_match(msg):
    matches = re.findall("((^|:)([0-9a-fA-F]{0,4})){1,8}$", msg)

    if len(matches) > 0:
        return True

    return False

def mac_address_match(msg):
    matches = re.findall("([0-9a-fA-F]{2}([:-]|$)){6}$|([0-9a-fA-F]{4}([.]|$)){3}", msg)

    if len(matches) > 0:
        return True

    return False

def password_match(msg):
    matches = re.findall("(?=^.{8,}$)((?=.*\d)|(?=.*\W+))(?![.\n])(?=.*[A-Z])(?=.*[a-z]).*$", msg)

    if len(matches) > 0:
        return True

    return False


if __name__ == "__main__":
    res = ipv6_match("4405:4380:")
    print(res)
