from tkinter import *
from tkinter import messagebox
from watchdog.observers import Observer
import service.my_handler
import service.db_utils
import service.usb_utils
import service.clipboard_data
import browserhistory as bh

import main

def main_process():
    window.withdraw()
    new_window = Toplevel(window)
    new_window.protocol("WM_DELETE_WINDOW", lambda: window.destroy())

    selected_path = dir_name.get()
    info_path = "curretn dir: " + str(selected_path)

    new_window.title("DLP")
    new_window.geometry("200x200")
    Label(new_window, text=info_path).pack()

    # event_handler = service.my_handler.MyHandler()
    # observer = Observer()
    #
    # observer.schedule(event_handler, path=selected_path, recursive=True)
    # observer.start()
    # service.db_utils.update_db()
    #
    # while True:
    #     service.usb_utils.check_all_drives()
    #     try:
    #         service.clipboard_data.get_data_from_clipboard()
    #     except KeyboardInterrupt:
    #         bh.write_browserhistory_csv()
    #         observer.stop()


window = Tk()
window.title('DLP')
window.geometry('400x300')

frame = Frame(
    window,
    padx=10,
    pady=10
)
frame.pack(expand=True)

dir_text = Label(
    frame,
    text="Введите директорию: "
)
dir_text.grid(row=3, column=1)

dir_name = Entry(frame,)
dir_name.grid(row=3, column=2, pady=5)

cal_btn = Button(
    frame,
    text='Рассчитать ИМТ',
    command=main_process
)
cal_btn.grid(row=5, column=2)

window.mainloop()