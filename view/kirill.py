import os

def remove_digits(array=None):
    new_arr = []

    for i in array:
        try:
            int(i)
            continue
        except ValueError:
            new_arr.append(i)

    return new_arr


if __name__ == "__main__":
    massive = ['dsadsad', '89', 'pop', '10', '921', 'dasd', 'ffff']
    n_arr = remove_digits(massive)
    print(n_arr)