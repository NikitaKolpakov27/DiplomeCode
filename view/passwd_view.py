import tkinter.messagebox
import sys
from tkinter import *
from service import passwd_utils, file_utils

def main_process(passwd_window):

    def on_closing():
        check_passwd()

    def return_to_base():
        passwd_window.destroy()

    def check_passwd():
        if passwd_utils.check_passwd(passwd_name.get()):
            passwd_window.destroy()
            file_utils.write_log("\nСистема     " + "Программа была выключена ")
            sys.exit("It's not the exception, program just finished")
        else:
            tkinter.messagebox.showerror("Bad Password", "Incorrect password!")
            file_utils.write_log("\nСистема     " + "Неправильный пароль для выхода из программы ")

    # passwd_window = Tk()
    passwd_window.title('Password')
    passwd_window.geometry('400x200')

    frame = Frame(passwd_window, padx=10, pady=10)
    frame.pack(expand=True)

    logo_text = Label(frame, text="DLP-система", font=("Helvetica", 18), foreground="blue")
    logo_text.grid(row=0, column=2)

    passwd_text = Label(frame, text="Введите пароль", font=("Helvetica", 14))
    passwd_text.grid(row=1, column=2)

    passwd_name = Entry(frame, width=20, font=("Helvetica", 14))
    passwd_name.grid(row=3, column=2, pady=5)

    passwd_btn = Button(frame, text='ОК', command=check_passwd, font=("Helvetica", 14))
    passwd_btn.grid(row=4, column=2)

    return_btn = Button(frame, text='Вернуться', command=return_to_base, font=("Helvetica", 14))
    return_btn.grid(row=5, column=2)

    passwd_window.protocol("WM_DELETE_WINDOW", on_closing)
    passwd_window.mainloop()


if __name__ == "__main__":
    main_process()
