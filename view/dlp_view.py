from tkinter import *
from tkinter import messagebox
from watchdog.observers import Observer
import service.my_handler
import service.db_utils
import service.usb_utils
import service.clipboard_utils
import browserhistory as bh

import main
from service import clipboard_utils
from view import view_utils


def main_process():
    window.withdraw()
    new_window = Toplevel(window)
    new_window.protocol("WM_DELETE_WINDOW", lambda: window.destroy())

    new_window.title("DLP. Main Process")
    new_window.geometry("600x400")

    selected_path = dir_name.get()
    info_path = "ТЕКУЩАЯ ДИРЕКТОРИЯ: " + str(selected_path) + "\n" + "==============" + "\n"

    main_log = Text(new_window, width=600, height=200, bg='black',
                    font=("Courier New", 12), foreground="white", wrap=WORD)
    main_log.pack()
    main_log.insert(1.0, info_path)

    ##########test-success!
    #view_utils.usb_check(main_log)
    ##########test

    event_handler = service.my_handler.MyHandler()
    observer = Observer()

    observer.schedule(event_handler, path=selected_path, recursive=True)
    observer.start()
    service.db_utils.update_db()

    # while True:
    #service.usb_utils.check_all_drives()
    view_utils.usb_check(main_log)



    view_utils.clipboard_info_view2(main_log, new_window)
    # new_window.after(0, clipboard_utils.get_data_from_clipboard2, new_window)
    new_window.mainloop()
    # try:
    #     # service.clipboard_utils.get_data_from_clipboard()
    #     view_utils.clipboard_info_view(main_log, new_window)
    #
    #     # new_window.after(1000, view_utils.clipboard_info_view)
    #     new_window.update()
    #     new_window.mainloop()
    # except KeyboardInterrupt:
    #     bh.write_browserhistory_csv()
    #     observer.stop()





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
    text='Начать процесс',
    command=main_process
)
cal_btn.grid(row=5, column=2)

window.mainloop()