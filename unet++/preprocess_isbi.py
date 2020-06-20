#用于resie图片和label
import os
from glob import glob
import cv2
from tqdm import tqdm

def main():
    img_size = 512

    # glod函数遍历该路径下的所有文件
    paths_train_img = glob('dataset/new train set/train_img/*')
    paths_train_label = glob('dataset/new train set/train_label/*')
    paths_test_img = glob("dataset/new_test_set/test_img/*")
    paths_test_label = glob("dataset/new_test_set/test_label/*")


    # 创建的目录名称，exist_ok=True,即使存在目录也不会报错
    os.makedirs('dataset/isbi_train_%d/train_img' % img_size, exist_ok=True)
    os.makedirs('dataset/isbi_train_%d/train_label' % img_size, exist_ok=True)
    os.makedirs('dataset/isbi_test_%d/test_img' % img_size, exist_ok=True)
    os.makedirs('dataset/isbi_test_%d/test_label' % img_size, exist_ok=True)


    #resize train_img
    for i in tqdm(range(len(paths_train_img))):
        path = paths_train_img[i]
        img = cv2.imread(path)
        img = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(os.path.join('dataset/isbi_train_%d/train_img' % img_size,
                                 os.path.basename(path)), img)
    #resize train_label
    for i in tqdm(range(len(paths_train_label))):
        if(i%2==1):
            pass
        else:
            path = paths_train_label[i]
            print(path)
            img = cv2.imread(path)
            img = cv2.resize(img, (img_size, img_size))
            cv2.imwrite(os.path.join('dataset/isbi_train_%d/train_label' % img_size,
                                     os.path.basename(path)), img)
    #resize test_img
    for i in tqdm(range(len(paths_test_img))):
        path = paths_test_img[i]
        img = cv2.imread(path)
        img = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(os.path.join('dataset/isbi_test_%d/test_img' % img_size,
                                 os.path.basename(path)), img)
    #resize test_label
    for i in tqdm(range(len(paths_test_label))):
        path = paths_test_label[i]
        img = cv2.imread(path)
        img = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(os.path.join('dataset/isbi_test_%d/test_label' % img_size,
                                 os.path.basename(path)), img)

if __name__ == '__main__':
    main()
