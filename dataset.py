import os
import argparse
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.utils.data as data


class ISBI(data.Dataset):

    def __init__(self, root, mode='train', transform=None, target_transform=None):
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

        if mode == 'train':
            self.train_data = []
            self.train_labels = []
            train_root = os.path.join(root, 'new train set')
            train_image_dir = os.path.join(train_root, 'train_img')
            train_label_dir = os.path.join(train_root, 'train_label')
            train_image_path = os.listdir(train_image_dir)
            train_label_path = os.listdir(train_label_dir)
            train_label_path = [
                i for i in train_label_path if i.endswith('.png')]
            # load train data
            for image_path in train_image_path:
                image_path = os.path.join(train_image_dir, image_path)
                image = Image.open(image_path)
                self.train_data.append(image)
            # load train label
            for label_path in train_label_path:
                label_path = os.path.join(train_label_dir, label_path)
                label = Image.open(label_path)
                self.train_labels.append(label)

        else:
            self.test_data = []
            self.test_labels = []
            test_root = os.path.join(root, 'new_test_set')
            test_image_dir = os.path.join(test_root, 'test_img')
            test_label_dir = os.path.join(test_root, 'test_label')
            test_image_path = os.listdir(test_image_dir)
            test_label_path = os.listdir(test_label_dir)
            # load test data
            for image_path in test_image_path:
                image_path = os.path.join(test_image_dir, image_path)
                image = Image.open(image_path)
                self.test_data.append(image)
            # load test label
            for label_path in test_label_path:
                label_path = os.path.join(test_label_dir, label_path)
                label = Image.open(label_path)
                self.test_labels.append(label)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.mode == 'train':
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        else:
            return len(self.test_data)


def return_data(args):
    root = args.root
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = ISBI(root=root, mode='train',
                      transform=transform, target_transform=transform)
    test_data = ISBI(root=root, mode='test',
                     transform=transform, target_transform=transform)

    train_loader = DataLoader(train_data,
                              batch_size=1,
                              shuffle=True,
                              num_workers=0,
                              drop_last=True)
    test_loader = DataLoader(test_data,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             drop_last=True)
    return train_loader, test_loader


if __name__ == "__main__":
    # load('../dataset')
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    train_loader, test_loader = return_data(args)

    for idx, (image, label) in enumerate(train_loader):
        print('image', image.shape)
        print('label', label.shape)

    for idx, (image, label) in enumerate(test_loader):
        print('image', image.shape)
        print('label', label.shape)
