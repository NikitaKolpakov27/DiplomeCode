from watchdog.observers import Observer
import clipboard_data
import db_utils
import my_handler


if __name__ == "__main__":
    event_handler = my_handler.MyHandler()
    observer = Observer()

    selected_path = input("Введите директорию: ")

    observer.schedule(event_handler, path=selected_path, recursive=True)
    observer.start()
    db_utils.update_db()

    print("Текущая директория: ", selected_path)
    print("==================================")

    while True:
        try:
            clipboard_data.get_data_from_clipboard()
        except KeyboardInterrupt:
            observer.stop()
