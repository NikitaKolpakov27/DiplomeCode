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
        main_log.insert(INSERT, "Флеш-накопитель(-и) был(-и) подключен(-ы)" + "\n")

        # Смена цвета
        main_log.tag_add("usb", 'end-2c linestart', 'end-2c')
        main_log.tag_config("usb", foreground="yellow")

        if has_a_conf_file > 0:
            main_log.insert(INSERT,
                            "На флешке (" + flash_dir + ") было обнаружено " + str(has_a_conf_file) + " конф. файлов."
                            + "\n")

            main_log.tag_add("usb", 'end-2c linestart', 'end-2c')
            main_log.tag_config("usb", foreground="red")
        else:
            main_log.insert(END, "На флешке (" + flash_dir + ") НЕ было обнаружено никаких конф. файлов." + "\n")

    else:
        main_log.insert(END, "Нет флеш-накопителей" + "\n")

        main_log.tag_add("usb", 'end-2c linestart', 'end-2c')
        main_log.tag_config("usb", foreground="yellow")


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

    clipboard_result = clipboard_utils.get_data_from_clipboard2(window, last_data)
    cl_data, cl_type, cl_event, last_data = next(clipboard_result)

    if cl_event == 'normal':

        if cl_type == 'file':
            main_log.insert(END, "\n" + "Буфер -> " + str(cl_data) + "(FILE)")
            # print("file normal")
        else:
            main_log.insert(END, "\n" + "Буфер -> " + str(cl_data) + "(TEXT)")
            # print("text normal")

    else:
        if cl_type == 'text':
            main_log.insert(END, "\n" + "Буфер -> " + str(cl_data) + "[This text may contain confidential data!!!]")
            # print("text conf")

        if cl_type == 'same data':
            main_log.insert(END, ".")
    window.after(1000, clipboard_info_view2, main_log, window)




if __name__ == "__main__":
    clipboard_info_view(1)

