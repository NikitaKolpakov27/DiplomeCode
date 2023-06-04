
# from tkinter import *
# import time
# win = Tk()
# win.title("tutorialspoint.com")
# win.geometry("600x400")
#
# op = 1
#
# def repeated():
#     global op
#     op += 1
#     my_lab.insert(END, str(op) + "\n")
#     win.after(1000, repeated)
#
# # def clock():
# #     hh= time.strftime("%I")
# #     mm= time.strftime("%M")
# #     ss = time.strftime("%S")
# #     day = time.strftime("%A")
# #     ap = time.strftime("%p")
# #     time_zone= time.strftime("%Z")
# #     my_lab.config(text=hh + ":" + mm + ":" + ss)
# #     my_lab.after(1000, clock)
# #     my_lab1.config(text=time_zone + " " + day)
# #
#
# my_lab= Text(win,font=("sans-serif", 26), fg="red")
# my_lab.pack(pady=20)
# # my_lab1= Label(win, text= "", font=("Helvetica",20), fg="blue")
# # my_lab1.pack(pady=10)
#
# repeated()
#
# win.mainloop()
import os
import random
import tkinter.messagebox

from service import conf_detect
from service.file_utils import get_file_type, read_pdf_file, read_docx_file, read_txt_file

if __name__ == "__main__":
    # from tkinter import *
    # from tkinter import ttk
    #
    # gui = Tk()
    # gui.title('Delftstack')
    # gui.geometry('600x400')
    #
    #
    # def StartProgress():
    #     # start progress
    #     progress_var.start(10)
    #
    #
    # def StopProgress():
    #     # stop progress
    #     progress_var.stop()
    #
    #
    # # create an object of progress bar
    # progress_var = ttk.Progressbar(gui, orient=HORIZONTAL, length=400, mode='indeterminate')
    # progress_var.pack(pady=30)
    # btn = Button(gui, text='progress', command=StartProgress)
    # btn.pack(pady=30)
    #
    # btn2 = Button(gui, text='stop', command=StopProgress)
    # btn2.pack(pady=30)
    # gui.mainloop()

    chars = 'abcdefghijklnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
    password = ''
    for i in range(10):
        password += random.choice(chars)
    print(password)
