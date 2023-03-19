from watchdog.observers import Observer
import clipboard_data
import my_handler
import tools


if __name__ == "__main__":
    event_handler = my_handler.MyHandler()
    observer = Observer()

    selected_path = "D:\\На новый ноут\\Учёба\\TEST FOLDER"

    observer.schedule(event_handler, path=selected_path, recursive=True)
    observer.start()
    tools.update_db()

    print("Текущая директория: ", selected_path)
    print("==================================")

    while True:
        try:
            clipboard_data.get_data_from_clipboard()
        except KeyboardInterrupt:
            observer.stop()
