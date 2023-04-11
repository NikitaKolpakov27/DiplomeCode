from watchdog.observers import Observer
import clipboard_data
import conf_detect
import db_utils
import my_handler
import usb_utils
import browserhistory as bh

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
        usb_utils.check_all_drives()
        try:
            clipboard_data.get_data_from_clipboard()
        except KeyboardInterrupt:
            bh.write_browserhistory_csv()
            observer.stop()