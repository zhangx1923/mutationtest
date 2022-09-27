import os

def print_msg(file, msg):
    print(msg)
    f = open(file, "a")
    f.write(msg)
    f.write("\r\n")
    f.close()


def create_floder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return True