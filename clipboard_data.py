import tkinter as tk
import tools

def get_data_from_clipboard():
    last_data = None

    conf_hashes = tools.get_conf_hashes()

    while True:

        root = tk.Tk()
        root.withdraw()
        data = root.clipboard_get()
        data_type = tools.is_string_path(data)

        if data != last_data and len(data) != 0:

            if data_type:

                print(data)

                if tools.hash_file(data) in conf_hashes:
                    print("WARNING!")
                    tools.conf_info_detected(data)
                    root.clipboard_clear()
                    break
            else:
                print(data)

        last_data = data


if __name__ == "__main__":
    pass
    # get_text_data_from_clipboard()
    # md5txt = hash_file("D:\\На новый ноут\\Учёба\\TEST FOLDER\\bithc.txt")
    # print("needed text: ", md5txt)
    #
    get_data_from_clipboard()
    # get_pic_from_clipboard()
