from watchdog.observers import Observer
import clipboard_data
import db_utils
import my_handler
from usb_utils import get_flash_directories, check_flash_drives
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
        flash_dirs = get_flash_directories()

        if len(flash_dirs) > 0:
            check_flash_drives(flash_dirs)

        try:
            clipboard_data.get_data_from_clipboard()
        except KeyboardInterrupt:
            bh.write_browserhistory_csv()
            observer.stop()
