import os
import numpy as np
from PIL import Image
import PIL.Image as Image
# tensorflow.keras.datasets import mnist
#from tensorflow.keras.datasets import fashion_mnist
#from tensorflow.keras.datasets import cifar10

def square(dig,perc_anom_train = 0.2,perc_anom_test = 0.2,size = 5,intensity = 'rand',DATASET = 'mnist', seed=None):
    '''
    :param dig: Selected dataset class
    :param perc_anom_train: Anomalies percentage in th training set (# of anomalies)/(# of sample in the training set)
    :param perc_anom_test: Anomalies percentage in th test set (# of anomalies)/(# of sample in the test set)
    :param size: Dimension of the square
    :param intensity: Pixel value for anomalous square
    :param DATASET: Dataset to use, possible choices: mnist, fmnist and cifar
    :return: X_train, Y_train, X_test, Y_test, GT_train, GT_test
    '''

    np.random.seed(seed=seed)

    if DATASET == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if DATASET == 'fmnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.copy()
        x_test = x_test.copy()
        y_train = y_train.copy()
        y_test = y_test.copy()
    if DATASET == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.


    width,height = x_train.shape[1:3]
    edge = size//2

    num_anomalies_train = int(perc_anom_train*(np.where(y_train==dig)[0].size))


    id_an_train = np.random.choice(np.where(y_train==dig)[0],num_anomalies_train, replace=False)
    GT_train = np.zeros((x_train.shape[0],width,height))

    pos_intensities = [0.2, 0.4, 0.6, 0.8]

    for id in id_an_train:
        center_x = np.random.randint(edge, width - edge)
        center_y = np.random.randint(edge, height - edge)
        if intensity == 'rand':
            #intens = np.random.randint(0,255,3)/255.
            c = np.random.randint(0, 4)
            intens = np.full(3, fill_value=pos_intensities[c])
        if len(x_train.shape) == 4:
            x_train[id, center_x - edge:center_x+edge+1,center_y-edge:center_y+edge+1,0] = intens[0]
            x_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1, 1] = intens[1]
            x_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1, 2] = intens[2]
        else:
            x_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = intens[0]

        GT_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = 1

    num_anomalies_test = int(perc_anom_test*(np.where(y_test==dig)[0].size))

    id_an_test = np.random.choice(np.where(y_test == dig)[0], num_anomalies_test, replace=False)
    GT_test = np.zeros((x_test.shape[0], width, height))


    for id in id_an_test:
        center_x = np.random.randint(edge, width - edge)
        center_y = np.random.randint(edge, height - edge)

        if intensity == 'rand':
            #intens = np.random.randint(0, 255, 3) / 255.
            c = np.random.randint(0, 4)
            intens = np.full(3, fill_value=pos_intensities[c])

        if len(x_test.shape) == 4:
            x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1, 0] = intens[0]
            x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1, 1] = intens[1]
            x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1, 2] = intens[2]
        else:
            x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = intens[0]


        GT_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = 1
    id_train = np.where(y_train == dig)
    id_test = np.where(y_test == dig)



    X_train = x_train[id_train]
    X_test = x_test[id_test]
    GT_train = GT_train[id_train]
    GT_test = GT_test[id_test]
    y_train = y_train.copy()
    y_train[id_train] = 0
    y_train[id_an_train] = 1
    Y_train = y_train[id_train]
    y_test = y_test.copy()
    y_test[id_test] = 0
    y_test[id_an_test] = 1
    Y_test = y_test[id_test]


    if len(X_test.shape) < 4:
        X_train = X_train.reshape(X_train.shape + (1,))
        X_test = X_test.reshape(X_test.shape + (1,))

    X_train = np.swapaxes(X_train, 2, 3)
    X_train = np.swapaxes(X_train, 1, 2)

    X_test = np.swapaxes(X_test, 2, 3)
    X_test = np.swapaxes(X_test, 1, 2)

    GT_train = GT_train.reshape(GT_train.shape[0], 1, GT_train.shape[1], GT_train.shape[2])
    GT_test = GT_test.reshape(GT_test.shape[0], 1, GT_test.shape[1], GT_test.shape[2])

    print(num_anomalies_train)
    print(num_anomalies_test)

    return X_train, Y_train, X_test, Y_test, GT_train, GT_test


def square_diff(dig,perc_anom_train = 0.2,perc_anom_test = 0.2,size = 5,intensity = 0.2,DATASET = 'mnist', seed=None):
    '''
    :param dig: Selected dataset class
    :param perc_anom_train: Anomalies percentage in th training set (# of anomalies)/(# of sample in the training set)
    :param perc_anom_test: Anomalies percentage in th test set (# of anomalies)/(# of sample in the test set)
    :param size: Dimension of the square
    :param intensity: Pixel value for anomalous square
    :param DATASET: Dataset to use, possible choices: mnist, fmnist and cifar
    :return: X_train, Y_train, X_test, Y_test, GT_train, GT_test
    '''

    np.random.seed(seed=seed)

    if DATASET == 'mnist' or DATASET == 'mnist_diff':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if DATASET == 'fmnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    if DATASET == 'cifar':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.


    width,height = x_train.shape[1:3]
    edge = size//2

    num_anomalies_train = int(perc_anom_train*(np.where(y_train==dig)[0].size))


    id_an_train = np.random.choice(np.where(y_train==dig)[0],num_anomalies_train, replace=False)
    GT_train = np.zeros((x_train.shape[0],width,height))


    for id in id_an_train:
        center_x = np.random.randint(edge, width - edge)
        center_y = np.random.randint(edge, height - edge)

        x_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = \
            np.where(x_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] + intensity <=1,
                     x_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] + intensity,
                     x_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] - intensity)

        GT_train[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = 1

    num_anomalies_test = int(perc_anom_test*(np.where(y_test==dig)[0].size))

    id_an_test = np.random.choice(np.where(y_test == dig)[0], num_anomalies_test, replace=False)
    GT_test = np.zeros((x_test.shape[0], width, height))


    for id in id_an_test:
        center_x = np.random.randint(edge, width - edge)
        center_y = np.random.randint(edge, height - edge)

        x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = \
            np.where(
                x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] + intensity <= 1,
                x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] + intensity,
                x_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] - intensity)


        GT_test[id, center_x - edge:center_x + edge + 1, center_y - edge:center_y + edge + 1] = 1
    id_train = np.where(y_train == dig)
    id_test = np.where(y_test == dig)



    X_train = x_train[id_train]
    X_test = x_test[id_test]
    GT_train = GT_train[id_train]
    GT_test = GT_test[id_test]
    y_train[id_train] = 0
    y_train[id_an_train] = 1
    Y_train = y_train[id_train]
    y_test[id_test] = 0
    y_test[id_an_test] = 1
    Y_test = y_test[id_test]


    if len(X_test.shape) < 4:
        X_train = X_train.reshape(X_train.shape + (1,))
        X_test = X_test.reshape(X_test.shape + (1,))

    X_train = np.swapaxes(X_train, 2, 3)
    X_train = np.swapaxes(X_train, 1, 2)

    X_test = np.swapaxes(X_test, 2, 3)
    X_test = np.swapaxes(X_test, 1, 2)

    GT_train = GT_train.reshape(GT_train.shape[0], 1, GT_train.shape[1], GT_train.shape[2])
    GT_test = GT_test.reshape(GT_test.shape[0], 1, GT_test.shape[1], GT_test.shape[2])

    print(num_anomalies_train)
    print(num_anomalies_test)

    return X_train, Y_train, X_test, Y_test, GT_train, GT_test

def mvtec(cl, path, n_anom_per_cls, seed=29):
    np.random.seed(seed=seed)

    labels = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )

    class_o = labels[cl]
    root = os.path.join(path, class_o)

    X_train = []
    X_test = []
    GT_train = []
    GT_test = []
    GT = []
    Y=[]

    #normal data
    f_path = os.path.join(root, 'train', 'good')
    normal_files_tr = os.listdir(f_path)
    for file in normal_files_tr:
        if file.lower().endswith(('png', 'jpg', 'npy')):
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB').resize((256, 256),Image.NEAREST))
            X_train.append(image)
            GT_train.append(np.zeros_like(image, dtype=np.uint8))
            GT.append(np.zeros_like(image, dtype=np.uint8))
            Y.append(0)
            X_test.append(image)
            GT_test.append(np.zeros_like(image, dtype=np.uint8))

    #anomalous data
    outlier_data_dir = os.path.join(root, 'test')
    outlier_classes = os.listdir(outlier_data_dir)
    for cl_a in outlier_classes:
        if cl_a == 'good':
            continue
        if cl_a == '.DS_Store':
            continue

        outlier_file = np.array(os.listdir(os.path.join(outlier_data_dir, cl_a)))
        idxs = np.random.permutation(len(outlier_file))

        for file in outlier_file[idxs[:n_anom_per_cls]]:
            if file.lower().endswith(('png', 'jpg', 'npy')):
                image = np.array(Image.open(os.path.join(root, 'test', cl_a, file)).convert('RGB').resize((256, 256),Image.NEAREST))
                #print("img shape: ", image.shape)
                X_train.append(image)
                GT_train.append(np.zeros_like(image, dtype=np.uint8))
                X_test.append(image)
                GT_test.append(np.zeros_like(image, dtype=np.uint8))
                Y.append(1)
                GT.append(np.array(Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/' + file).replace('.png', '_mask.png')).convert('RGB').resize((256, 256),Image.NEAREST)))

    X_train = np.array(X_train).astype(np.uint8)
    X_test = np.array(X_test).astype(np.uint8)

    GT=np.array(GT)
    GT_train = np.array(GT_train)
    GT_test = np.array(GT_test)

    Y=np.array(Y)

    Y_train = np.zeros(X_train.shape[0])
    Y_test = np.zeros(X_test.shape[0])

    return X_train, Y_train, GT_train, X_test, Y_test, GT_test, GT, Y



def load_brainMRI_dataset(path, img_size=(256, 256), seed=None):
    np.random.seed(seed=seed)

    train_img_path = os.path.join(path, 'train')
    test_img_path = os.path.join(path, 'train')

    X_train = []
    X_test = []
    GT_train = []
    GT_test = []

    normal_files_tr = os.listdir(train_img_path)
    for file in normal_files_tr:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = Image.open(os.path.join(train_img_path, file)).convert('RGB')
            image = image.resize(img_size)
            X_train.append(np.array(image))
            GT_train.append(np.zeros(np.array(image).shape[:2], dtype=np.uint8))

    normal_files_te = os.listdir(test_img_path)
    for file in normal_files_te:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = Image.open(os.path.join(test_img_path, file)).convert('RGB')
            image = image.resize(img_size)
            X_test.append(np.array(image))
            GT_test.append(np.zeros(np.array(image).shape[:2], dtype=np.uint8))

    X_train = np.array(X_train).astype(np.uint8)
    X_test = np.array(X_test).astype(np.uint8)
    Y_train = np.zeros(X_train.shape[0])
    Y_test = np.zeros(X_test.shape[0])

    return X_train, Y_train, GT_train, X_test, Y_test, GT_test