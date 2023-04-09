import os
import psutil as psutil
import conf_utils
import file_utils

# Проверка подключенных флеш-накопителей
def check_all_drives():
    flash_dirs = get_flash_directories()
    if len(flash_dirs) > 0:
        print("Флеш-накопитель(-и) был(-и) подключен(-ы)")
        check_flash_drives(flash_dirs)
    else:
        print("Нет флеш-накопителей")


# Получить директории для флеш-накопителей
def get_flash_directories():
    drives = psutil.disk_partitions()
    flash_dirs = []

    for i in drives:

        if str(i[2]).__contains__("FAT") and str(i[3]) == "rw,removable":
            flash_dirs.append(i[0])

    return flash_dirs


# Проверка всех подключённых флеш-накопителей на конф. файлы
def check_flash_drives(flash_dirs):

    for flash_dir in flash_dirs:
        conf_files_count = 0

        for root, dirs, files in os.walk(os.path.abspath(flash_dir)):
            for filename in files:

                file_type = file_utils.get_file_type(filename)

                if file_type == ".docx" or file_type == ".pdf" or file_type == ".txt":
                    real_path = os.path.join(root, filename)
                    conf_res = conf_utils.is_file_or_text_confidential(False, real_path)

                    if conf_res == True:
                        conf_files_count += 1

        if conf_files_count > 0:
            print("On flash drive (" + flash_dir + ") was detected " + str(conf_files_count) + " confidential files.")
        else:
            print("On flash drive (" + flash_dir + ")" + " was haven't been detected any confidential files.")


if __name__ == "__main__":
    fl_dr = get_flash_directories()
    print(fl_dr)

    check_flash_drives(fl_dr)

    # print('C диск информация:', d[0])
    # print('D информация о диске:', d[1])
    # print('Информация о диске:', d[2])
    # print('Получить поле диска:', d[0][0], d[1][0], d[2][0])
    # print('Тип данных:', type(d), '\n')
