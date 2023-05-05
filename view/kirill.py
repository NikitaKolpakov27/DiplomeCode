import os

if __name__ == "__main__":
    cur_path = os.path.dirname(__file__)
    new_path = os.path.relpath('..\\dataset\\normal\\ba', cur_path)

    with open(new_path, "r") as f:
        res = f.read()

    print(cur_path)