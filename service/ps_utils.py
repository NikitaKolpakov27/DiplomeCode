import time
import psutil
from service.file_utils import write_log

current_process_list = []
new_process_list = []
def get_processes(flag, secs):
    global new_process_list
    global current_process_list

    while flag:
        time.sleep(secs)

        for process in psutil.process_iter():
            log_process = f"{process.pid}: {process.name()}"
            new_process_list.append(log_process)

        if not new_process_list == current_process_list:
            str_list = "\n\t\t\t".join(new_process_list)

            # Запись в лог
            write_log("\n" + "Приложения     " + str_list)
            current_process_list = new_process_list


if __name__ == "__main__":
    get_processes(True, 5)
