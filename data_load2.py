import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from imgaug import augmenters as iaa
from augment import train_augementors, infer_augmentors
import imgaug as ia
from sklearn.model_selection import StratifiedKFold
import cv2
root1 = r'C:\Users\suer0426\Desktop\Data\new_data_3.15'
root2 = r'C:\Users\suer0426\Desktop\Data\new_data_3.28(benign)'
k_fold=3

class ToTensor(object):

    """
    This is a transform(augmentation)class
    convert ndarrays in sample to Tensors
    """

    # swap color axis because
    # input : numpy image: H x W x C
    # output: torch image: C X H X W

    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


class DatasetTrain(Dataset):
    def __init__(self, x_sub, y_sub, augment_list):
        self.x_sub = x_sub
        self.y_sub = y_sub
        self.length = len(self.x_sub)
        self.to_tenser_augment = augment_list[0]
        self.shape_agument = augment_list[1]
        self.color_augment = augment_list[2]

    def __getitem__(self, item):
        '''item is mask index'''
        image = cv2.imread(self.x_sub[item])
        #image = image[..., ::-1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.shape_agument(image)
        image = self.color_augment(image)
        image = self.to_tenser_augment(image)
        y_class = self.y_sub[item]
        return (image, y_class)

    def __len__(self):
        return self.length


class DatasetVal(Dataset):
    def __init__(self, x_sub, y_sub, augment_list):
        self.x_sub = x_sub
        self.y_sub = y_sub
        self.length = len(self.x_sub)
        self.to_tenser_augment = augment_list[0]
        self.infer_augment = augment_list[1]

    def __getitem__(self, item):
        '''item is mask index'''
        image = cv2.imread(self.x_sub[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.infer_augment(image)
        image = self.to_tenser_augment(image)
        y_class = self.y_sub[item]
        return (image, y_class)

    def __len__(self):
        return self.length


def print_dataset_info(x_whole_np, y_whole_np, sub_idx, phase = 'train'):

    """
    input : x_whole_np -> ndarray contains type string of paths of image file
            y_whole_np -> ndarray contains type integer of y_index(targets)
            sub_idx -> numpy array contains candidate indexes
            transform -> object of certain class that object[object_of_DatasetClass[i]] returns transformed item

    return : object of newly made DatasetClass adress
    """
    #For sanity check, print out the number of dataset in each classes.
    y0_index = sub_idx[y_whole_np[sub_idx]==0]
    y1_index = sub_idx[y_whole_np[sub_idx]==1]
    y2_index = sub_idx[y_whole_np[sub_idx]==2]
    y3_index = sub_idx[y_whole_np[sub_idx]==3]

    count = 0
    for item in x_whole_np[sub_idx]:
        if 'new_data_3.28(benign)' in item:
            count += 1
    if phase == 'train':
        print('--------train dataset info---------')
    else:
        print('--------val dataset info---------')

    print('benign from new_data_3.28 :', count)
    print('benign from new_data_3.15 :', len(y0_index) - count)
    print('yo_num :{}\n'.format(len(y0_index)))
    print('y1_num :{}\n'.format(len(y1_index)))
    print('y2_num :{}\n'.format(len(y2_index)))
    print('y3_num :{}\n'.format(len(y3_index)))


# make whole dataset list
# input : root that path
# output : x_whole, y_whole that contains all file paths and classes each
def make_dataset():
    x_whole = []
    y_whole = []

    for (path, dir, filenames) in os.walk(root1):
        for filename in filenames:
            file_path = os.path.join(path, filename)
            y_class = int(file_path[-5])
            # if y_class != 0: for binary classification
            #     y_class = 1
            x_whole.append(file_path)
            y_whole.append(y_class)

    for (path, dir, filenames) in os.walk(root2):
        for filename in filenames:
            file_path = os.path.join(path, filename)
            y_class = 0
            x_whole.append(file_path)
            y_whole.append(y_class)

    print('x_whole length : {}, y_whole length : {}'.format(len(x_whole), len(y_whole)))
    print('x_whole is {}'.format(x_whole))
    print('y_whole is {}'.format(y_whole))

    print('#yclass1 : {}, #yclass2 : {}, #yclass3:{}, #yclass4:{}'.
          format(np.sum(np.asarray(y_whole) == 0),
                 np.sum(np.asarray(y_whole) == 1),
                 np.sum(np.asarray(y_whole) == 2),
                 np.sum(np.asarray(y_whole) == 3))
          )



    '''
    
    x_whole: ['1th filepath.png', '2nd filepath.png'. '3rd filepath.png' ...] <- list that contains all image_paths
    y_whole: [1                    2                   0                 ...] <- list contains all y(target) values
    k_fold:  divide items into k-folds each which contains validation data and training data    
    output : dataset list , for example, if k_fold = 3, list length of 3 [{'train' : 1st_train_dataset_obj, 'val':1st_val_dataset_obj} ...]
    
    
    '''
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state =1)

    dataset = []

    x_whole_np = np.asarray(x_whole)
    y_whole_np = np.asarray(y_whole)
    #y0_index = sub_idx[y_whole_np[sub_idx]==0]

    toTenser_aug = ToTensor()
    aug_train_list = [toTenser_aug,
                      iaa.Sequential(train_augementors()[0]).augment_image,
                      iaa.Sequential(train_augementors()[1]).augment_image]
    aug_val_list = [toTenser_aug,
                    iaa.Sequential(infer_augmentors()).augment_image]

    for train_idx, val_idx in skf.split(x_whole, y_whole):
        '''
        a little bit tweak of train_idx, val_idx,
        first, there are serious imbalance between classes numbers in both train_idx, val_idx,
        so we need to fix that..
        Especially, in terms of val_idx, copy the validation set is not acceptable, so I will cut and put it to train_idx
        '''
        y0_train_idx = train_idx[y_whole_np[train_idx] == 0]
        y1_train_idx = train_idx[y_whole_np[train_idx] == 1]
        y2_train_idx = train_idx[y_whole_np[train_idx] == 2]
        y3_train_idx = train_idx[y_whole_np[train_idx] == 3]


        i = 0
        while len(y0_train_idx) < len(y2_train_idx):
            y0_train_idx = np.append(y0_train_idx, y0_train_idx[i])
            i += 1

        i = 0
        while len(y1_train_idx) < len(y2_train_idx):
            y1_train_idx = np.append(y1_train_idx, y1_train_idx[i])
            i += 1
        i = 0

        # while len(y2_train_idx) < len(y0_train_idx):
        #     y2_train_idx = np.append(y2_train_idx, y2_train_idx[i])
        #     i += 1
        # i = 0

        while len(y3_train_idx) < len(y2_train_idx):
            y3_train_idx = np.append(y3_train_idx, y3_train_idx[i])
            i += 1

        train_idx = np.concatenate((y0_train_idx, y1_train_idx, y2_train_idx, y3_train_idx), axis=None)

        np.random.shuffle(train_idx)

        x_sub_train = x_whole_np[train_idx]
        y_sub_train = y_whole_np[train_idx]
        x_sub_val = x_whole_np[val_idx]
        y_sub_val = y_whole_np[val_idx]

        print_dataset_info(x_whole_np, y_whole_np, train_idx, phase='train')
        print_dataset_info(x_whole_np, y_whole_np, val_idx, phase='val')

        dataset_train = DatasetTrain(x_sub_train, y_sub_train, aug_train_list)
        dataset_val = DatasetVal(x_sub_val, y_sub_val, aug_val_list)

        dataset.append((dataset_train, dataset_val))
    return dataset






    '''
    train_augs = [shape_augs, input_augs]. 
    To use train_augs,
    # shape must be deterministic so it can be reused
        shape_augs = self.shape_augs.to_deterministic()
        input_img = shape_augs.augment_image(input_img) 
    # color augmentation 
            input_img = self.input_augs.augment_image(input_img)
    '''