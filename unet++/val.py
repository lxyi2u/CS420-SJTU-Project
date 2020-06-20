import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()
    #name是模型所在文件夹名
    parser.add_argument('--name', default='unet++_100epoch_0.01_5batchsize',
                        help='model name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))

    print('-'*20)
    cudnn.benchmark = True

    # 创建模型
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # 加载测试数据
    test_img_ids = glob(os.path.join('dataset', config['dataset_test'], 'test_img', '*' + config['img_ext']))
    test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_img_ids]

    val_img_ids = test_img_ids

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('dataset', config['dataset_test'], 'test_img'),
        mask_dir=os.path.join('dataset', config['dataset_test'], 'test_label'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # 进行预测
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)
            #计算iou loss，作为衡量模型效果的一部分
            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()


            for i in range(len(output)):
                for c in range(config['num_classes']):
                    #将图片转为二值化，转为黑白图片
                    output[i, c] = output[i, c] > 0.6
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meter.avg)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
