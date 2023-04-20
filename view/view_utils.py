import tkinter.messagebox
from tkinter import *
import service.my_handler
import service.db_utils
import service.usb_utils
import service.clipboard_utils
from service import clipboard_utils

last_data = None

def usb_check(main_log):
    usb_checking_result = service.usb_utils.check_all_drives()

    has_a_flash_drive = usb_checking_result[0]
    has_a_conf_file = usb_checking_result[1]
    flash_dir = usb_checking_result[2]

    if has_a_flash_drive:
        main_log.insert(INSERT, "\n" + "Флеш-накопитель(-и) был(-и) подключен(-ы)")

        # Смена цвета
        main_log.tag_add("usb_begin", 'end-2c linestart', 'end-2c')
        main_log.tag_config("usb_begin", foreground="yellow")

        if has_a_conf_file > 0:
            main_log.insert(INSERT,
                    "\n" + "На флешке (" + flash_dir + ") было обнаружено " + str(has_a_conf_file) + " конф. файлов.")

            main_log.tag_add("usb_conf", 'end-2c linestart', 'end-2c')
            main_log.tag_config("usb_conf", foreground="red")
        else:
            main_log.insert(END, "\n" + "На флешке (" + flash_dir + ") НЕ было обнаружено никаких конф. файлов.")

    else:
        main_log.insert(END, "\n" + "Нет флеш-накопителей")

        main_log.tag_add("usb_no", 'end-2c linestart', 'end-2c')
        main_log.tag_config("usb_no", foreground="yellow")


def clipboard_info_view(main_log, window):
    clipboard_result = clipboard_utils.get_data_from_clipboard2(window)

    for cl_data, cl_type, cl_event in clipboard_result:

        if cl_event == 'normal':

            if cl_type == 'file':
                main_log.insert(END, "Буфер -> " + str(cl_data) + "(FILE)" + "\n")
                # print("file normal")
            else:
                main_log.insert(END, "Буфер -> " + str(cl_data) + "(TEXT)" + "\n")
                # print("text normal")

        else:
            if cl_type == 'text':
                main_log.insert(END, "Буфер -> " + str(cl_data) + "[This text may contain confidential data!!!]" + "\n")
                # print("text conf")

def clipboard_info_view2(main_log, window):
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

            main_log.tag_add("clip_file", 'end-2c linestart', 'end-2c')
            main_log.tag_config("clip_file", foreground="#1959d1")
        else:
            main_log.insert(END, "\n" + "Буфер -> " + str(cl_data) + " (TEXT)")

            main_log.tag_add("clip_text", 'end-2c linestart', 'end-2c')
            main_log.tag_config("clip_text", foreground="#42aaff")

    else:
        if cl_type == 'text':
            main_log.insert(END, "\n" + "Буфер -> " + str(cl_data))
            main_log.insert(END, "[This text may contain confidential data!!!]")

            main_log.tag_add("clip_conf", 'end-2c linestart', 'end-2c')
            main_log.tag_config("clip_conf", foreground="red")

        if cl_type == 'file':
            main_log.insert(END, "\n" + "||||||WARNING!|||||")
            main_log.tag_add("clip_conf", 'end-2c linestart', 'end-2c')
            main_log.tag_config("clip_conf", foreground="red")

            err_message = cl_data
            tkinter.messagebox.showwarning(title="Опасность", message=err_message)

        if cl_type == 'same data':
            main_log.insert(END, ".")
    window.after(1000, clipboard_info_view2, main_log, window)


def handler_info_view(main_log, event, ev_type, status, path):
    if ev_type == 'file':

        if status == 'normal':
            main_log.insert(END, "\n" + event + " file -- " + path)

            main_log.tag_add("monitor", 'end-2c linestart', 'end-2c')
            main_log.tag_config("monitor", foreground="violet")

        if status == 'conf':
            main_log.insert(END, "\n" + event + " file (CONFIDENTIAL) -- " + path)

            main_log.tag_add("monitor", 'end-2c linestart', 'end-2c')
            main_log.tag_config("monitor", foreground="red")

    else:
        main_log.insert(END, "\n" + event + " directory -- " + path)

        main_log.tag_add("monitor", 'end-2c linestart', 'end-2c')
        main_log.tag_config("monitor", foreground="blue")


if __name__ == "__main__":
    pass

