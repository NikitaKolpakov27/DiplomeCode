import os
import psutil as psutil
import service.conf_utils as conf_utils
import service.file_utils as file_utils


def check_all_drives():
    """
        Проверка подключенных флеш-накопителей

        :return:
            * Если подключены -> проверка их на наличие конфиденциальных файлов
            * Если нет, то ничего
    """
    flash_dirs = get_flash_directories()

    if len(flash_dirs) > 0:
        print("Флеш-накопитель(-и) был(-и) подключен(-ы)")
        return check_flash_drives(flash_dirs)
    else:
        print("Нет флеш-накопителей")
        return False, 0, None


def get_flash_directories():
    """
        Получение директории для флеш-накопителей

        :return: list flash_dirs: список доступных директорий флеш-накопителей
    """
    drives = psutil.disk_partitions()
    flash_dirs = []

    for i in drives:

        if str(i[2]).__contains__("FAT") and str(i[3]) == "rw,removable":
            flash_dirs.append(i[0])

    return flash_dirs


def check_flash_drives(flash_dirs):
    """
        Проверка всех подключённых флеш-накопителей на конф. файлы

        :param list flash_dirs: список доступных директорий флеш-накопителей
        :return: кортеж, состоящий из информации о файлах; нужно для GUI
    """

    # Цикл для каждой директории
    for flash_dir in flash_dirs:

        # Счетчик конфиденциальных файлов
        conf_files_count = 0

        # Список конфиденциальных файлов (для записи в отдельный лог)
        conf_files_list = []

        # Проходимся по всем файлам в текущей директории
        for root, dirs, files in os.walk(os.path.abspath(flash_dir)):
            for filename in files:

                # Получаем тип (разрешение) файла
                file_type = file_utils.get_file_type(filename)

                # Рассматриваем только доступные типы (docx, pdf и txt)
                if file_type == ".docx" or file_type == ".pdf" or file_type == ".txt":

                    # Получаем полный путь файла
                    real_path = os.path.join(root, filename)

                    # Проверяем, есть ли в тексте файла признаки конфиденциального характера
                    conf_res = conf_utils.is_file_or_text_confidential(False, real_path)

                    # Если есть -> увеличиваем счетчик
                    if conf_res:
                        conf_files_count += 1
                        conf_files_list.append(real_path)

        if conf_files_count > 0:
            print("On flash drive (" + flash_dir + ") was detected " + str(conf_files_count) + " confidential files.")
            file_utils.write_log("USB     На накопителе (" + flash_dir + ") было замечено " + str(conf_files_count) + " конфиденциальных файлов:" + str(conf_files_list), '..\\Reports\\usb_reports')
            return True, conf_files_count, flash_dir
        else:
            print("On flash drive (" + flash_dir + ")" + " was haven't been detected any confidential files.")
            return True, 0, flash_dir
