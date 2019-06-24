import os
import numpy as np

root = '/home/hyunwoo/Desktop/new_data'

def print_dataset_info(root):
    # make whole dataset list
    # input : root that path
    # output : x_whole, y_whole that contains all file paths and classes each

    x_whole = []
    y_whole = []

    for(path, dir, filenames) in os.walk(root):
        for filename in filenames:
            file_path = os.path.join(path, filename)
            y_class = int(file_path[-5])
            x_whole.append(file_path)
            y_whole.append(y_class)

    print('x_whole length : {}, y_whole length : {}'.format(len(x_whole), len(y_whole)))
    print('x_whole is {}'.format(x_whole))
    print('y_whole is {}'.format(y_whole))

    print('#yclass1 : {}, #yclass2 : {}, #yclass3 : {} #yclass4 : {}'.format(np.sum(np.asarray(y_whole) == 0),
                                                  np.sum(np.asarray(y_whole) == 1),
                                                  np.sum(np.asarray(y_whole) == 2),
                                                  np.sum(np.asarray(y_whole) == 3)))


print_dataset_info(root)