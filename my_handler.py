from watchdog.events import FileSystemEventHandler
import tools


class MyHandler(FileSystemEventHandler):
    def on_any_event(self, event):

        if not event.is_directory:
            conf_file = tools.is_file_or_text_confidential(False, event.src_path)

            if conf_file:
                print(event.event_type, " file (CONFIDENTIAL) -- ", event.src_path)
            else:
                print(event.event_type, " file -- ", event.src_path)
        else:
            print(event.event_type, " directory -- ", event.src_path)

    def on_created(self, event):
        pass

    def on_deleted(self, event):
        pass

    def on_modified(self, event):
        pass

    def on_moved(self, event):
        pass

    def on_closed(self, event):
        pass
