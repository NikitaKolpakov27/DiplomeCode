from tkinter import *
import service.my_handler
import service.db_utils
import service.usb_utils
import service.clipboard_utils
from service import clipboard_utils, db_utils
from service.file_utils import write_log

last_data = None
usb_message = None


def usb_check(main_log):
    """
        Вывод информации в GUI по поводу операций с USB накопителями и их файлами

        :param main_log: основной лог (куда идут все сообщения)
        :return: None
    """

    global usb_message  # Мб убрать. Больше нигде не используется. Посмотреть, что произойдет дальше

    # Получение информации из USB-портов
    usb_checking_result = service.usb_utils.check_all_drives()

    has_a_flash_drive = usb_checking_result[0]
    has_a_conf_file = usb_checking_result[1]
    flash_dir = usb_checking_result[2]

    # Проверка наличия USB-носителя вообще
    if has_a_flash_drive:
        main_log.insert(INSERT, "\n" + "Флеш-накопитель(-и) был(-и) подключен(-ы)")

        # Запись в лог
        write_log("\n" + "USB     " + "Флеш-накопитель(-и) был(-и) подключен(-ы)")

        # Смена цвета
        main_log.tag_add("usb_begin", 'end-2c linestart', 'end-2c')
        main_log.tag_config("usb_begin", foreground="yellow")

        # Результат проверки USB-носителя на наличие конфиденциальных файлов
        if has_a_conf_file > 0:
            main_log.insert(INSERT, "\n" + "На флеш-накопителе (" + flash_dir + ") было обнаружено " + str(
                                has_a_conf_file) + " конф. файлов.")
            # Запись в лог
            write_log("\n" + "USB     " + "На флеш-накопителе (" + flash_dir + ") было обнаружено " + str(
                                has_a_conf_file) + " конф. файлов.")

            main_log.tag_add("usb_conf", 'end-2c linestart', 'end-2c')
            main_log.tag_config("usb_conf", foreground="red")
        else:
            main_log.insert(END,
                            "\n" + "На флеш-накопителе (" + flash_dir + ") НЕ было обнаружено никаких конф. файлов.")
            # Запись в лог
            write_log("\n" + "USB     " + "На флеш-накопителе (" + flash_dir + ") НЕ было обнаружено никаких конф. файлов.")

    else:
        main_log.insert(END, "\n" + "Нет флеш-накопителей")
        # Запись в лог
        write_log("\n" + "USB     " + "Нет флеш-накопителей")

        main_log.tag_add("usb_no", 'end-2c linestart', 'end-2c')
        main_log.tag_config("usb_no", foreground="yellow")

def clipboard_info_view(main_log, window):
    global last_data

    try:
        clipboard_result = clipboard_utils.get_data_from_clipboard2(window, last_data)
        cl_data, cl_type, cl_event, last_data = next(clipboard_result)
    except StopIteration:
        cl_data = None
        cl_type = 'same data'
        cl_event = 'other'

    if cl_event == 'normal':

        if cl_type == 'file':
            main_log.insert(END, "\n" + "Буфер -> " + str(cl_data) + " (FILE)")
            # Запись в лог
            write_log("\n" + "Буфер     " + str(cl_data).replace("\n", "") + " (FILE)")

            main_log.tag_add("clip_file", 'end-2c linestart', 'end-2c')
            main_log.tag_config("clip_file", foreground="#1959d1")
        else:
            main_log.insert(END, "\n" + "Буфер -> " + str(cl_data) + " (TEXT)")
            # Запись в лог
            write_log("\n" + "Буфер     " + str(cl_data).replace("\n", "") + " (TEXT)")

            main_log.tag_add("clip_text", 'end-2c linestart', 'end-2c')
            main_log.tag_config("clip_text", foreground="#42aaff")

    else:
        if cl_type == 'text':
            main_log.insert(END, "\n" + "Буфер -> " + str(cl_data))
            main_log.insert(END, "[This text may contain confidential data!!!]")
            # Запись в лог
            write_log("\n" + "Буфер     " + str(cl_data) + "[This text may contain confidential data!!!]")

            main_log.tag_add("clip_conf", 'end-2c linestart', 'end-2c')
            main_log.tag_config("clip_conf", foreground="red")

        if cl_type == 'file':
            main_log.insert(END, "\n" + "||||||WARNING!|||||")
            # Запись в лог
            write_log("\n" + "||||||WARNING!|||||")
            main_log.tag_add("clip_conf", 'end-2c linestart', 'end-2c')
            main_log.tag_config("clip_conf", foreground="red")

            err_message = cl_data

            # Пока уберем, чтоб не мешало
            # tkinter.messagebox.showwarning(title="Опасность", message=err_message)

        if cl_type == 'same data':
            main_log.insert(END, ".")
    window.after(1000, clipboard_info_view, main_log, window)


def handler_info_view(main_log, event, ev_type, status, path, message):
    """
        Вывод информации в GUI по поводу операций с файлами или директориями

        :param main_log: основной лог (куда идут все сообщения)
        :param event: само событие
        :param ev_type: тип события (file, dir)
        :param status: статус события (normal, conf)
        :param path: путь к файлу, в котором / (с которым) произошло событие
        :param message: сообщение (при конфиденциальном статусе => при диалоговом окне)

        :return: None
    """

    # Проверяем тип события (с файлом)
    if ev_type == 'file':

        # Проверяем статус события (нормальный)
        if status == 'normal':
            main_log.insert(END, "\n" + event + " file -- " + path)
            # Запись в лог
            write_log("\n" + str(event).upper() + " FILE     " + path)

            main_log.tag_add("monitor_norm", 'end-2c linestart', 'end-2c')
            main_log.tag_config("monitor_norm", foreground="violet")

        # Если файл - конфиденциальный
        if status == 'conf':
            main_log.insert(END, "\n" + event + " file (CONF) -- " + path)
            # Запись в лог
            write_log("\n" + str(event).upper() + " FILE (CONF)     " + path)

            # Пока уберем
            # tkinter.messagebox.showerror(title="Опасность", message=message)

            main_log.tag_add("monitor_conf", 'end-2c linestart', 'end-2c')
            main_log.tag_config("monitor_conf", foreground="red")

            # Обновляем БД
            db_utils.update_db()

    # Если тип события с директорией
    else:
        main_log.insert(END, "\n" + event + " directory -- " + path)
        # Запись в лог
        write_log("\n" + str(event).upper() + " DIRECTORY     " + path)

        main_log.tag_add("monitor_dir", 'end-2c linestart', 'end-2c')
        main_log.tag_config("monitor_dir", foreground="blue")


if __name__ == "__main__":
    pass
