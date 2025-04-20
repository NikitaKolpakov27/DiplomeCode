import os.path
import subprocess
import tkinter.messagebox
from tkinter import *
from tkinter import filedialog, ttk
import tkinter.tix

from watchdog.observers import Observer
import service.my_handler
import service.db_utils
import service.usb_utils
import service.clipboard_utils
import browserhistory as bh

from service import conf_detect, bayes, net_utils, mail, ai_conf
from service.file_utils import get_file_type, read_docx_file, read_pdf_file, read_txt_file
from view import view_utils, passwd_view


event_handler = service.my_handler.MyHandler()
observer = Observer()

def check_file_for_conf():
    window.withdraw()
    conf_window = Toplevel(window)
    conf_window.protocol("WM_DELETE_WINDOW", lambda: window.destroy())

    conf_window.title("DLP. Checking file for confidential features")
    conf_window.geometry("500x200")

    conf_label = Label(conf_window, text="Проверяется файл: ")
    conf_label.pack()

    text = ""
    path_to_file = filedialog.askopenfilename()
    # print('Selected:', path_to_file)

    conf_file_label = Label(conf_window, text=path_to_file)
    conf_file_label.pack()

    progress_var = ttk.Progressbar(conf_window, orient=HORIZONTAL, length=400, mode='indeterminate')
    progress_var.pack(pady=30)
    progress_var.start(10)

    # Проверка типа файла
    file_type = get_file_type(path_to_file)
    if file_type == ".pdf":
        text = read_pdf_file(path_to_file)

    elif file_type == ".docx":
        text = read_docx_file(path_to_file)

    elif file_type == ".txt":
        text = read_txt_file(path_to_file)

    else:
        tkinter.messagebox.showerror(title="Ошибка", message="Файлы такого типа пока что не поддерживаются :(")
        window.destroy()

    # percentage_conf = conf_detect.check_conf_info(text)
    # progress_var.stop()
    # msg = "Данный файл содержит признаков информации ограниченного доступа на " + str(percentage_conf) + "%"
    # tkinter.messagebox.showinfo(title="Результат проверки", message=msg)
    conf_result = bayes.bayes_text_classify(test_text=text)

    if conf_result:
        msg = "Данный файл содержит признаков информации ограниченного доступа"
        tkinter.messagebox.showinfo(title="Результат проверки", message=msg)
    else:
        msg = "Данный файл НЕ содержит признаков информации ограниченного доступа"
        tkinter.messagebox.showinfo(title="Результат проверки", message=msg)

    conf_window.mainloop()


def check_mail():
    window.withdraw()
    mail_window = Toplevel(window)
    mail_window.protocol("WM_DELETE_WINDOW", lambda: window.destroy())

    def receiving():
        mail_login.grid_remove()
        mail_passwd.grid_remove()
        mail_password.grid_remove()
        mail_name.grid_remove()
        mail_process.grid_remove()

        process_bar = Label(mail_window, text="Идет процесс получения.", font=("Helvetica", 14))
        process_bar.grid(row=1, column=1, pady=20)

        progress_var = ttk.Progressbar(mail_window, orient=HORIZONTAL, length=400, mode='indeterminate')
        progress_var.grid(row=2, column=1, pady=30)
        progress_var.start(10)

        mail.get_sent_emails(mail_login=mail_login.get(), mail_passwd=mail_passwd.get())
        tkinter.messagebox.showinfo(title="E-mail messages receiving", message="Отправленные письма были получены!")

    mail_window.title("DLP. Checking e-mail letters")
    mail_window.geometry("600x200")

    mail_name = Label(mail_window, text="Введите название ящика: ", font=("Helvetica", 14))
    mail_name.grid(row=1, column=1)

    mail_login = Entry(mail_window, width=30, font=("Helvetica", 14))
    mail_login.grid(row=1, column=2, pady=5)

    mail_password = Label(mail_window, text="Введите пароль от внешнего ящика", font=("Helvetica", 14))
    mail_password.grid(row=2, column=1)

    mail_passwd = Entry(mail_window, width=30, font=("Helvetica", 14))
    mail_passwd.grid(row=2, column=2, pady=5)

    mail_process = Button(mail_window, text='Проверить почту', command=receiving, font=("Helvetica", 14))
    mail_process.grid(row=3, column=1)

    mail_window.mainloop()


def check_text_ai():
    def click_classify():
        result_label.configure(text="Идет проверка...")
        result_label.configure(text=ai_conf.classify_view_version(input_message.get("1.0", END)))

    window.withdraw()
    ai_window = Toplevel(window)
    ai_window.protocol("WM_DELETE_WINDOW", lambda: window.destroy())

    ai_window.title("DLP. Checking text by AI")
    ai_window.geometry("600x300")

    write_message = Label(ai_window, text="Введите ваше сообщение: ", font=("Helvetica", 14))
    write_message.grid(column=0, row=0)

    input_message = Text(ai_window, width=60, height=5)
    input_message.grid(column=0, row=1)

    classify_button = Button(ai_window, text="Получить результат", command=click_classify, font=("Helvetica", 14))
    classify_button.grid(column=0, row=2)

    result_label = Label(ai_window, font=("Helvetica", 14), foreground="blue")
    result_label.grid(column=0, row=4)

    ai_window.mainloop()

def main_process():

    # Проверка валидности введенного пути
    selected_path = dir_name.get()
    # Временно
    # selected_path = "d:\\test folder"
    check_path = os.path.isdir(selected_path)

    if not check_path:

        if os.path.isfile(selected_path):
            tkinter.messagebox.showwarning(title="Неверный путь",
                        message="Ввод: " + str(selected_path) + "\nВведен путь к файлу, а не к директории!")
        else:
            tkinter.messagebox.showwarning(title="Неверный путь",
                        message="Ввод: " + str(selected_path) + "\nСистеме не удается найти указанный путь!")

    else:

        def window_passwd():
            # new_window.withdraw()
            passwd_window = Toplevel(new_window)
            # passwd_window.protocol("WM_DELETE_WINDOW", lambda: new_window.destroy())

            passwd_view.main_process(passwd_window)

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

        # Добавление скроллбара
        scrollbar = Scrollbar(new_window, orient="vertical", command=main_log.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        # scrollbar = tkinter.tix.ScrolledWindow(new_window, 600, 400)
        # scrollbar.pack()

        # Получение информации об окне для обсервера
        event_handler.get_info(main_log, new_window)

        # Старт обсервера
        observer.schedule(event_handler, path=selected_path, recursive=True)
        observer.start()

        # Обновление БД
        service.db_utils.update_db()
        service.db_utils.make_conf_file_db()

        # Главный процесс
        try:
            view_utils.usb_check(main_log)
            # net_utils.disable_websites()
            view_utils.clipboard_info_view(main_log, new_window)
            new_window.protocol("WM_DELETE_WINDOW", window_passwd)
            new_window.mainloop()
        except KeyboardInterrupt:
            # net_utils.enable_websites()
            bh.write_browserhistory_csv()
            observer.stop()

window = Tk()
window.title('DLP')
window.geometry('715x200')

frame = Frame(window, padx=10, pady=10)
frame.pack(expand=True)

logo_text = Label(frame, text="DLP-система", font=("Helvetica", 18), foreground="blue")
logo_text.grid(row=0, column=2)

dir_text = Label(frame, text="Введите директорию: ", font=("Helvetica", 14))
dir_text.grid(row=1, column=1)

dir_name = Entry(frame, width=30, font=("Helvetica", 14))
dir_name.grid(row=1, column=2, pady=5)

dlp_btn = Button(frame, text='Начать процесс', command=main_process, font=("Helvetica", 14))
dlp_btn.grid(row=1, column=3)

check_conf_btn = Button(frame, text='Проверить файл', command=check_file_for_conf, font=("Helvetica", 14))
check_conf_btn.grid(row=5, column=1)

check_mail_btn = Button(frame, text='Проверить почту', command=check_mail, font=("Helvetica", 14))
check_mail_btn.grid(row=5, column=2)

check_text_ai = Button(frame, text='Классификация AI', command=check_text_ai,
                       font=("Helvetica", 14), foreground="purple")
check_text_ai.grid(row=5, column=3)

window.mainloop()
