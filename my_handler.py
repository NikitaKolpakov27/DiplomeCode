from watchdog.events import FileSystemEventHandler
import conf_utils
import db_utils
import usb_utils


class MyHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        usb_utils.check_all_drives()

        if not event.is_directory:
            conf_file = conf_utils.is_file_or_text_confidential(False, event.src_path)

            if conf_file:
                conf_utils.conf_info_detected(event.src_path, event.event_type)
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
        db_utils.update_db()

    def on_moved(self, event):
        pass

    def on_closed(self, event):
        pass
