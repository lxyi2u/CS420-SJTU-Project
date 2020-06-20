import argparse
import torch
from trainer import Trainer
from fcn16s import FCN16s
from fcn8s import FCN8s
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
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # dataset
    train_loader, test_loader = return_data(args)

    # model
    model = FCN8s(n_class=2)
    start_epoch = 0
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.name)
        model.load_state_dict(checkpoint)
    else:
        fcn16s = FCN16s(n_class=2)
        state_dict = torch.load(args.fcn16)
        try:
            fcn16s.load_state_dict(state_dict)
        except RuntimeError:
            fcn16s.load_state_dict(state_dict['model_state_dict'])
        model.copy_params_from_fcn16s(fcn16s)
    if cuda:
        model = model.cuda()

    # optimizer
    if args.optimizer == 'SGD':
        optim = torch.optim.SGD(
            get_parameters(model, bias=False),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optim = torch.optim.Adam(
            get_parameters(model, bias=False),
            lr=args.lr,
            betas=(0.9, 0.999))

    # train
    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=200,
        size_average=False,
        name=args.name,
        loss=args.loss
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    if args.mode == 'train':
        trainer.train()
    else:
        trainer.draw()

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
                        default=1.0e-6, help='learning rate')
    parser.add_argument('--weight-decay', type=float,
                        default=0.0005, help='weight decay')
    parser.add_argument('--momentum', type=float,
                        default=0.99, help='momentum')
    parser.add_argument('--root', type=str, default='../dataset/membrane')
    parser.add_argument('--out', type=str, default='../out')
    parser.add_argument('--name', type=str, default='fcn8s')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--loss', type=str, default='focal')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--fcn16', type=str, default='fcn16s.pkl')
    args = parser.parse_args()
    main(args)
