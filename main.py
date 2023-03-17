from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import clipboard_data
import tools


class MyHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        print()

        if not event.is_directory:
            conf_file = tools.is_file_confidential(event.src_path)

            if conf_file:
                print(event.event_type, " file (CONFIDENTIAL) -- ", event.src_path)
            else:
                print(event.event_type, " file -- ", event.src_path)
        else:
            print(event.event_type, " directory -- ", event.src_path)

    def on_created(self, event):
        pass
        # print("on_created", event.src_path)

    def on_deleted(self, event):
        pass
        # print("on_deleted", event.src_path)

    def on_modified(self, event):
        # print("on_modified", event.src_path)
        pass

    def on_moved(self, event):
        pass
        # print("on_moved", event.src_path)

    def on_closed(self, event):
        pass


if __name__ == "__main__":
    event_handler = MyHandler()
    observer = Observer()

    selected_path = "D:\\"

    observer.schedule(event_handler, path=selected_path, recursive=True)
    observer.start()

    print("Текущая директория: ", selected_path)
    print("==================================")

    while True:
        try:
            clipboard_data.get_data_from_clipboard()
        except KeyboardInterrupt:
            observer.stop()
