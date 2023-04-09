import tkinter as tk
import conf_utils
import file_utils


def get_data_from_clipboard():
    data = ""
    last_data = None

    conf_hashes = conf_utils.get_conf_hashes()

    while True:

        root = tk.Tk()
        root.withdraw()

        try:
            data = root.clipboard_get()
        except Exception as e:
            if e == "CLIPBOARD selection doesn't exist or form \"STRING\" not defined":
                print("Clipboard is empty")
                break

        data_type = file_utils.is_string_path(data)

        if data != last_data and len(data) != 0:

            # Проверка, является ли текст в буфере путем к файлу или нет
            if data_type:

                print("Буфер -> ", data, "(FILE)")

                if file_utils.hash_file(data) in conf_hashes:
                    print("WARNING!")
                    conf_utils.conf_info_detected(data, "Copy")
            else:

                conf_res = conf_utils.is_file_or_text_confidential(True, data)

                if conf_res:
                    print("Буфер -> ", data, "[This text may contain confidential data!!!]")
                else:
                    print("Буфер -> ", data, "TEXT")

        last_data = data


if __name__ == "__main__":
    get_data_from_clipboard()
