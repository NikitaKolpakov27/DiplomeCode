import PySimpleGUI as sg
from watchdog.observers import Observer
import browserhistory as bh
import service.my_handler
import service.db_utils
import service.usb_utils
import service.clipboard_data

event_handler = service.my_handler.MyHandler()
observer = Observer()


def main_process(selected_path):

    observer.schedule(event_handler, path=selected_path, recursive=True)
    observer.start()
    service.db_utils.update_db()

    elem = window['output_field']
    dir_info = "Текущая директория: " + str(selected_path) + "\n" + "=================================="
    elem.update(dir_info)

    while True:
        service.usb_utils.check_all_drives()
        try:
            service.clipboard_data.get_data_from_clipboard()
        except KeyboardInterrupt:
            bh.write_browserhistory_csv()
            observer.stop()


sg.theme("lightPurple")

layout = [
    [sg.Text('Введите директорию'), sg.InputText()],
    [sg.Output(size=(88, 20), key='output_field')],
    [sg.Submit(), sg.Cancel()]
]

window = sg.Window('DLP-system (demo)', layout)

while True:
    event, values = window.read()
    if event in (None, 'Exit', 'Cancel'):
        break

    if event == 'Submit':
        selected_path = values[0]
        main_process(selected_path)

window.close()
