import tkinter as tk
import service.conf_utils as conf_utils
import service.file_utils as file_utils

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

                # print("Буфер -> ", data, "(FILE)")

                # message = "Буфер -> " + str(data) + "(FILE)"
                yield data, 'file', 'normal'

                if file_utils.hash_file(data) in conf_hashes:
                    print("WARNING!")
                    yield None, 'file', 'warning'
                    conf_utils.conf_info_detected(data, "Copy")
            else:

                conf_res = conf_utils.is_file_or_text_confidential(True, data)

                if conf_res:
                    print("Буфер -> ", data, "[This text may contain confidential data!!!]")
                    yield data, 'text', 'conf'
                else:
                    print("Буфер -> ", data, "(TEXT)")
                    yield data, 'text', 'normal'

        last_data = data

def get_data_from_clipboard2(window, last_data):

    data = ""
    # last_data = None

    conf_hashes = conf_utils.get_conf_hashes()
    conf_files = conf_utils.get_conf_files()

    # root = tk.Tk()
    # root.withdraw()

    try:
        # data = root.clipboard_get()
        data = window.clipboard_get()
    except Exception as e:
        if e == "CLIPBOARD selection doesn't exist or form \"STRING\" not defined":
            print("Clipboard is empty")
            return

    data_type = file_utils.is_string_path(data)

    if data != last_data and len(data) != 0:

        # Проверка, является ли текст в буфере путем к файлу или нет
        if data_type:

            # Проверка на исключение (когда может податься на вход директория, что-то другое, а не файл)
            hash_result = ""
            try:
                hash_result = file_utils.hash_file(data)
            except Exception as e:
                message = str(e)
                yield message, 'file', 'warning', last_data

            if hash_result in conf_hashes:
                print("WARNING!")
                message = conf_utils.conf_info_detected(data, "Copy")
                last_data = data
                yield message, 'file', 'warning', last_data

            else:
                last_data = data
                yield data, 'file', 'normal', last_data

        else:

            conf_res = conf_utils.is_file_or_text_confidential(True, data)

            if conf_res:
                print("Буфер -> ", data, "[This text may contain confidential data!!!]")

                last_data = data
                yield data, 'text', 'conf', last_data
            else:
                print("Буфер -> ", data, "(TEXT)")

                last_data = data
                yield data, 'text', 'normal', last_data

    elif data == last_data:
        yield None, 'same data', None, last_data

    window.after(1000, get_data_from_clipboard2, window, last_data)


if __name__ == "__main__":
    get_data_from_clipboard2()
