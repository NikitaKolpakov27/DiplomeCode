
from tkinter import *
import time
win = Tk()
win.title("tutorialspoint.com")
win.geometry("600x400")

op = 1

def repeated():
    global op
    op += 1
    my_lab.insert(END, str(op) + "\n")
    win.after(1000, repeated)

# def clock():
#     hh= time.strftime("%I")
#     mm= time.strftime("%M")
#     ss = time.strftime("%S")
#     day = time.strftime("%A")
#     ap = time.strftime("%p")
#     time_zone= time.strftime("%Z")
#     my_lab.config(text=hh + ":" + mm + ":" + ss)
#     my_lab.after(1000, clock)
#     my_lab1.config(text=time_zone + " " + day)
#

my_lab= Text(win,font=("sans-serif", 26), fg="red")
my_lab.pack(pady=20)
# my_lab1= Label(win, text= "", font=("Helvetica",20), fg="blue")
# my_lab1.pack(pady=10)

repeated()

win.mainloop()