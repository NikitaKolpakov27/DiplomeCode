import tkinter as tk
import tools

def get_data_from_clipboard():
    data = ""
    last_data = None

    conf_hashes = tools.get_conf_hashes()

    while True:

        root = tk.Tk()
        root.withdraw()

        try:
            data = root.clipboard_get()
        except Exception as e:
            if e == "CLIPBOARD selection doesn't exist or form \"STRING\" not defined":
                print("Clipboard is empty")
                break

        data_type = tools.is_string_path(data)

        if data != last_data and len(data) != 0:

            # Проверка, является ли текст в буфере путем к файлу или нет
            if data_type:

                print("Буфер -> ", data)

                if tools.hash_file(data) in conf_hashes:
                    print("WARNING!")
                    tools.conf_info_detected(data, "Copy")
                    root.clipboard_clear()
                    break
            else:

                conf_res = tools.is_file_or_text_confidential(True, data)

                if conf_res:
                    print("Буфер -> ", "[This text has confidential data!!!]")
                else:
                    print("Буфер -> ", data)

        last_data = data


if __name__ == "__main__":
    get_data_from_clipboard()
