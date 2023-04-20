import os.path
import tkinter.messagebox
from tkinter import *
from watchdog.observers import Observer
import service.my_handler
import service.db_utils
import service.usb_utils
import service.clipboard_utils
import browserhistory as bh
from view import view_utils

event_handler = service.my_handler.MyHandler()
observer = Observer()

def main_process():
    # Проверка введенного пути
    selected_path = dir_name.get()
    check_path = os.path.isdir(selected_path)

    if not check_path:
        tkinter.messagebox.showwarning(title="Неверный путь", message="Системе не удается найти указанный путь!")
        window.destroy()

    else:

        # Удаление старого окна
        window.withdraw()
        new_window = Toplevel(window)
        new_window.protocol("WM_DELETE_WINDOW", lambda: window.destroy())

        # Создание нового окна
        new_window.title("DLP. Main Process")
        new_window.geometry("600x400")
        info_path = "ТЕКУЩАЯ ДИРЕКТОРИЯ: " + str(selected_path) + "\n" + "=====================" + "\n"

        main_log = Text(new_window, width=600, height=200, bg='black',
                        font=("Courier New", 12), foreground="white", wrap=WORD)
        main_log.pack()
        main_log.insert(1.0, info_path)

        # Получение информации об окне для обсервера
        event_handler.get_info(main_log, new_window)

        # Старт обсервера
        observer.schedule(event_handler, path=selected_path, recursive=True)
        observer.start()
        service.db_utils.update_db()

        # Главный процесс
        try:
            view_utils.usb_check(main_log)
            view_utils.clipboard_info_view2(main_log, new_window)
            new_window.mainloop()
        except KeyboardInterrupt:
            bh.write_browserhistory_csv()
            observer.stop()


window = Tk()
window.title('DLP')
window.geometry('600x200')

frame = Frame(window, padx=10, pady=10)
frame.pack(expand=True)

logo_text = Label(frame, text="DLP-система", font=("Helvetica", 18), foreground="blue")
logo_text.grid(row=0, column=2)

dir_text = Label(frame, text="Введите директорию: ", font=("Helvetica", 14))
dir_text.grid(row=1, column=1)

dir_name = Entry(frame, width=30, font=("Helvetica", 14))
dir_name.grid(row=1, column=2, pady=5)

cal_btn = Button(frame, text='Начать процесс', command=main_process, font=("Helvetica", 14))
cal_btn.grid(row=5, column=2)

window.mainloop()
