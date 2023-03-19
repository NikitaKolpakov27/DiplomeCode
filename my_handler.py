from watchdog.events import FileSystemEventHandler
import tools


class MyHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        print()

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

        if tools.is_file_or_text_confidential(False, event.src_path):
            print("|||WARNING||| This file is CONFIDENTIAL and it has been deleted!!!")
            tools.conf_info_detected(event.src_path, "Deleting")

    def on_modified(self, event):

        if tools.is_file_or_text_confidential(False, event.src_path):
            print("|||WARNING||| This file is CONFIDENTIAL and it has been modified!!!")
            tools.conf_info_detected(event.src_path, "Modifying")

    def on_moved(self, event):

        if tools.is_file_or_text_confidential(False, event.src_path):
            print("|||WARNING||| This file is CONFIDENTIAL and it has been moved!!!")
            tools.conf_info_detected(event.src_path, "Moving")

    def on_closed(self, event):
        pass