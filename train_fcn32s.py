import argparse
import torch
from trainer import Trainer
from fcn32s import FCN32s
from vgg import VGG16
from dataset import return_data


def main(args):
    cuda = torch.cuda.is_available()
    # dataset
    train_loader, test_loader = return_data(args)
    # model
    model = FCN32s(n_class=2)
    start_epoch = 0
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        vgg16 = VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)
    if cuda:
        model = model.cuda()

    # 3. optimizer
    optim = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=test_loader,
        out=args.out,
        max_iter=args.max_iteration,
        interval_validate=4000,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--resume', type=bool, default=False, help='checkpoint path')
    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    parser.add_argument('--max-iteration', type=int,
                        default=100000, help='max iteration')
    parser.add_argument('--lr', type=float,
                        default=1.0e-10, help='learning rate')
    parser.add_argument('--weight-decay', type=float,
                        default=0.0005, help='weight decay')
    parser.add_argument('--momentum', type=float,
                        default=0.99, help='momentum')
    parser.add_argument('--root', type=str, default='../dataset')
    parser.add_argument('--out', type=str, default='../out')
    args = parser.parse_args()
    main(args)
