def get_passwd():
    with open("./passwd") as f:
        passwd = f.read()

    return passwd

def check_passwd(entered_passwd):
    right_passwd = get_passwd()

    return entered_passwd == right_passwd
