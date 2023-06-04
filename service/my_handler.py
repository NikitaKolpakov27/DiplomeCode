from watchdog.events import FileSystemEventHandler
import service.conf_utils as conf_utils
import service.db_utils as db_utils
import view.view_utils
from service import passwd_utils


class MyHandler(FileSystemEventHandler):

    main_log = None
    window = None

    def on_any_event(self, event):
        view.view_utils.usb_check(self.main_log)
        passwd_utils.update_passwd()

        if not event.is_directory:
            conf_file = conf_utils.is_file_or_text_confidential(False, event.src_path)

            if conf_file:
                message = conf_utils.conf_info_detected(event.src_path, event.event_type)
                print(event.event_type, " file (CONFIDENTIAL) -- ", event.src_path)
                view.view_utils.handler_info_view(self.main_log,
                                                  str(event.event_type), 'file', 'conf', str(event.src_path), message)
                db_utils.update_conf_db()
            else:
                print(event.event_type, " file -- ", event.src_path)
                view.view_utils.handler_info_view(self.main_log,
                                                  str(event.event_type), 'file', 'normal', str(event.src_path), "")
        else:
            print(event.event_type, " directory -- ", event.src_path)
            view.view_utils.handler_info_view(self.main_log,
                                              str(event.event_type), 'dir', 'normal', str(event.src_path), "")

    def on_created(self, event):
        db_utils.update_db()

    def on_deleted(self, event):
        db_utils.update_db()
        # pass

    def on_modified(self, event):
        db_utils.update_db()
        # pass

    def on_moved(self, event):
        db_utils.update_db()

    def on_closed(self, event):
        pass

    def get_info(self, main_log, window):
        self.main_log = main_log
        self.window = window



