import os

from watchdog.observers import Observer
import service.my_handler
import service.db_utils
import service.usb_utils
import service.clipboard_utils
import browserhistory as bh

def main():
    event_handler = service.my_handler.MyHandler()
    observer = Observer()

    selected_path = input("Введите директорию: ")

    observer.schedule(event_handler, path=selected_path, recursive=True)
    observer.start()
    service.db_utils.update_db()

    print("Текущая директория: ", selected_path)
    print("==================================")

    while True:
        service.usb_utils.check_all_drives()
        try:
            service.clipboard_utils.get_data_from_clipboard()
        except KeyboardInterrupt:
            bh.write_browserhistory_csv()
            observer.stop()


if __name__ == "__main__":
    main()

    # Создание файлов с текстом из норм и конф файлов (пдф и ворд)
    # cur_path = os.path.dirname(__file__)
    # correct_path = os.path.relpath("..\\view", cur_path)
    # file_path = correct_path + "/о про.txt"
    #
    # file_nice = ""
    # with open(file_path, "r", encoding="utf-8") as file:
    #     file_nice = file.readlines()
    #
    # file_nice = [s.strip() for s in file_nice]
    # file_nice = " ".join(file_nice)
    #
    # new_file_path = correct_path + "/new_опро.txt"
    # with open(new_file_path, "a+", encoding="utf-8") as file:
    #     file.write(file_nice)
