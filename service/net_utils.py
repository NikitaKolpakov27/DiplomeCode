import os

redirect = "127.0.0.1"
# website_list = ["www.google.com", "www.twitch.tv"]
website_list = []
host_path = "C:\\Windows\\System32\\drivers\\etc\\hosts"

def disable_websites():

    # Добавление "запрещенных" сайтов в массив
    cur_path = os.path.dirname(__file__)
    ws_path = os.path.relpath("..\\view\\website_list", cur_path)

    f = open(ws_path, "r")
    web_sites = f.readlines()

    for i in web_sites:
        website_list.append(i.strip())
    f.close()

    file = open(host_path, "r+")
    content = file.read()
    for website in website_list:
        if website in content:
            pass
        else:
            file.write(redirect + " " + website + "\n")

def enable_websites():
    file = open(host_path, 'r+')
    content = file.readlines()
    file.seek(0)
    for line in content:
        if not any(website in line for website in website_list):
            file.write(line)
        file.truncate()
