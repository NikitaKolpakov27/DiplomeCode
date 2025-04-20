import datetime
import time

while True:
    with open("test_file.txt", "a+") as f:
        f.write(str(datetime.datetime.now()) + " Pop\n")
    f.close()
    time.sleep(2)
