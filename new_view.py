from tkinter import *
from tkinter import messagebox


def openNewWindow():
    window.withdraw()
    newWindow = Toplevel(window)
    newWindow.protocol("WM_DELETE_WINDOW", lambda: window.destroy())

    newWindow.title("New Window")
    newWindow.geometry("200x200")
    Label(newWindow,
          text="This is a new window").pack()

def calculate_bmi():

    try:
        kg = int(weight_tf.get())
        m = int(height_tf.get()) / 100
        bmi = kg / (m * m)
        bmi = round(bmi, 1)

        if bmi < 18.5:
            messagebox.showinfo('bmi-pythonguides', f'ИМТ = {bmi} соответствует недостаточному весу')
        elif (bmi > 18.5) and (bmi < 24.9):
            messagebox.showinfo('bmi-pythonguides', f'ИМТ = {bmi} соответствует нормальному весу')
        elif (bmi > 24.9) and (bmi < 29.9):
            messagebox.showinfo('bmi-pythonguides', f'ИМТ = {bmi} соответствует избыточному весу')
        else:
            messagebox.showinfo('bmi-pythonguides', f'ИМТ = {bmi} соответствует ожирению')
    except Exception:
        messagebox.showinfo('bmi-pythonguides', 'Что-то не так!')


window = Tk()
window.title('Калькулятор индекса массы тела (ИМТ)')
window.geometry('400x300')

frame = Frame(
    window,
    padx=10,
    pady=10
)
frame.pack(expand=True)

height_lb = Label(
    frame,
    text="Введите свой рост (в см)  "
)
height_lb.grid(row=3, column=1)

weight_lb = Label(frame, text="Введите свой вес (в кг)  ",)
weight_lb.grid(row=4, column=1)

height_tf = Entry(frame,)
height_tf.grid(row=3, column=2, pady=5)

weight_tf = Entry(
    frame,
)
weight_tf.grid(row=4, column=2, pady=5)

cal_btn = Button(
    frame,
    text='Рассчитать ИМТ',
    command=openNewWindow
)
cal_btn.grid(row=5, column=2)

window.mainloop()