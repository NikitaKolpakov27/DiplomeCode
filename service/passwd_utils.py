import random

def create_passwd():
    chars = 'abcdefghijklnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
    password = ''
    for i in range(10):
        password += random.choice(chars)
    return password

def update_passwd():
    open("./passwd", 'w').close()
    with open("./passwd", "w") as f:
        f.write(create_passwd())

def get_passwd():
    with open("./passwd") as f:
        passwd = f.read()

    return passwd

def check_passwd(entered_passwd):
    right_passwd = get_passwd()

    return entered_passwd == right_passwd
