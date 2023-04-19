from watchdog.observers import Observer
import service.my_handler
import service.db_utils
import service.usb_utils
import service.clipboard_data
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
            service.clipboard_data.get_data_from_clipboard()
        except KeyboardInterrupt:
            bh.write_browserhistory_csv()
            observer.stop()


if __name__ == "__main__":
    main()
