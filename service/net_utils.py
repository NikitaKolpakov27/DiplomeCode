import os
import time
from datetime import datetime as dt

redirect = "127.0.0.1"
website_list = ["www.google.com", "www.twitch.tv"]

cur_path = os.path.dirname(__file__)
host_path = os.path.relpath("..\\hosts", cur_path)

while True:

    website_list.append("www.ya.ru")
    print(website_list)

    if dt(dt.now().year, dt.now().month, dt.now().day, 8) < dt.now() \
            < dt(dt.now().year, dt.now().month, dt.now().day, 23):
        print("Rihanna")

        file = open(host_path, "r+")
        content = file.read()
        for website in website_list:
            if website in content:
                pass
            else:
                file.write(redirect + " " + website + "\n")
    else:
        print("Drake")

        file = open(host_path, 'r+')
        content = file.readlines()
        file.seek(0)
        for line in content:
            if not any(website in line for website in website_list):
                file.write(line)
            file.truncate()
    time.sleep(5)