import argparse
import torch
from trainer import Trainer
from fcn32s import FCN32s
from fcn16s import FCN16s
from vgg import VGG16
from dataset import return_data
import torch.nn as nn


def get_parameters(model, bias=False):

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        # elif isinstance(m, modules_skipped):
        #     continue
        # else:
        #     raise ValueError('Unexpected module: %s' % str(m))


def main(args):
    cuda = torch.cuda.is_available()
    # dataset
    train_loader, test_loader = return_data(args)
    # model
    model = FCN32s(n_class=2)
    start_epoch = 0
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.name)
        model.load_state_dict(checkpoint)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # start_epoch = checkpoint['epoch']
        # start_iteration = checkpoint['iteration']
    else:
        vgg16 = VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)
    if cuda:
        model = model.cuda()

    # 3. optimizer
    # optim = torch.optim.SGD(
    #     get_parameters(model, bias=False),
    #     lr=args.lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay)
    optim = torch.optim.Adam(
        get_parameters(model, bias=False),
        lr=args.lr,
        betas=(0.9, 0.999))
    # if args.resume:
    #     optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=200,
        size_average=False,
        name=args.name
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()
    # trainer.draw()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--resume', type=bool,
                        default=False, help='checkpoint path')
    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    parser.add_argument('--max-iteration', type=int,
                        default=100000, help='max iteration')
    parser.add_argument('--lr', type=float,
                        default=1.0e-5, help='learning rate')
    parser.add_argument('--weight-decay', type=float,
                        default=0.0005, help='weight decay')
    parser.add_argument('--momentum', type=float,
                        default=0.99, help='momentum')
    parser.add_argument('--root', type=str, default='../dataset')
    parser.add_argument('--out', type=str, default='../out')
    parser.add_argument('--name', type=str, default='fcn')
    args = parser.parse_args()
    main(args)
