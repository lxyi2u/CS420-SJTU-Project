import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from distutils.version import LooseVersion
from torchvision import transforms

def focal_loss(inputs, targets, gamma=2, focal_loss_alpha=0.5):

    n, c, h, w = inputs.size()
    inputs = F.softmax(inputs, dim=1)
    # log_p: (n*h*w, c)
    inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()
    inputs = inputs.view(-1, c)
    # target: (n*h*w,)
    targets = targets.transpose(1, 2).transpose(2, 3).contiguous()
    targets = targets.view(-1, 1)
    # print(inputs.shape)
    # print(targets.shape)
    # 计算正负样本权重
    alpha_factor = torch.ones(targets.shape).cuda() * focal_loss_alpha
    alpha_factor = torch.where(
        torch.eq(targets, 1), alpha_factor, 1. - alpha_factor)
    # 计算因子项
    focal_weight = torch.where(
        torch.eq(targets, 1), 1. - inputs, inputs).cuda()
    # 得到最终的权重
    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
    targets = targets.type(torch.FloatTensor).cuda()
    # 计算标准交叉熵
    # print(focal_weight.shape)
    # print(targets.shape)
    # print(inputs.shape)
    # bce = torch.log(inputs)
    # print(bce.shape)
    bce = -(targets * torch.log(inputs) +
            (1 - targets) * torch.log(1 - inputs))
    # exit()
    # focal loss
    cls_loss = focal_weight * bce
    return cls_loss.sum()


def cross_entropy2d(pred, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, c, h, w)
    n, c, h, w = pred.size()
    # print(pred.shape)
    # print(target.shape)
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(pred)
    else:
        # >=0.3
        log_p = F.log_softmax(pred, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    target = target.transpose(1, 2).transpose(2, 3).contiguous()
    target = target.view(-1)
    # print(log_p.shape)
    # print(target.shape)
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= (h*w)
    return loss


class Trainer(object):

    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, epochs,
                 size_average=False, name='fcn'):
        self.cuda = cuda
        self.model = model
        self.optim = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.size_average = size_average
        self.name = name
        self.epoch = 0
        self.iteration = 0

    def train_epoch(self):
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            # optimizer step
            # print(data[0,0,0,:100])
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            assert self.model.training
            self.optim.zero_grad()
            score = self.model(data)
            # loss = cross_entropy2d(score, target,
            #                        size_average=self.size_average)
            loss = focal_loss(score, target)
            loss /= len(data)
            loss_data = loss.data.item()
            loss.backward()
            self.optim.step()
            if iteration % 5 == 0:
                print('iteration:{} loss:{:.4f}'.format(iteration, loss_data))

        train_acc, train_loss, train_tacc, train_facc = self.validate(
            self.train_loader)
        # print(train_acc)
        test_acc, test_loss, test_tacc, test_facc = self.validate(
            self.val_loader, True)
        print('epoch:{} train accuracy:{:.4f} test accuracy:{:.4f}'.format(
            self.epoch, train_acc, test_acc))
        print('train loss:{:.4f} test loss:{:.4f}'.format(
            train_loss, test_loss))
        print('train tacc:{:.4f} train facc:{:.4f}'.format(
            train_tacc, train_facc))
        print('test tacc:{:.4f} test facc:{:.4f}'.format(
            test_tacc, test_facc))
        print(' ')
        # exit()

    def train(self):

        for epoch in range(self.epochs):
            self.epoch = epoch
            self.train_epoch()
        torch.save(self.model.state_dict(), '{}.pkl'.format(self.name))

    def validate(self, data_loader, verbose=False):
        training = self.model.training
        self.model.eval()

        with torch.no_grad():
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            accuracy = 0
            total_num = 0
            loss = 0
            for batch_idx, (data, target) in enumerate(data_loader):

                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                # score: [n, c, h, w]
                score = self.model(data)
                pred = F.softmax(score, dim=1).max(1)[1]
                loss += cross_entropy2d(score, target,
                                        size_average=self.size_average)

                # if verbose:
                #     if batch_idx == 0:
                #         # print(score)
                #         # print(pred)
                #         print(score[0, :, 200, 200:300])
                #         print(target[0, 0, 200, 200:300])
                #         print(pred.shape)
                #         print(pred[0, 200, 200:300])
                #         # print(pred.shape)
                #         # print(pred.float().mean())
                #         pass
                pred = pred.view(-1)
                target = target.view(-1)
                accuracy += torch.eq(pred, target).float().mean()
                total_num += data.shape[0]
                target_pt = target[pred == 1]
                target_pf = target[pred == 0]
                TP += target_pt.sum()
                FP += (target_pt.shape[0] - target_pt.sum())
                FN += target_pf.sum()
                TN += (target_pf.shape[0] - target_pf.sum())

                # print(TP)
                # print(total_num)
            avg_acc = accuracy / total_num
            total_loss = loss / total_num
            t_acc = TP.float() / (TP + FN).float()
            f_acc = TN.float() / (FP + TN).float()

        if training:
            self.model.train()

        return avg_acc.data.item(), total_loss.data.item(), t_acc.data.item(), f_acc.data.item()

    def draw(self):
        training = self.model.training
        self.model.eval()
        unloader = transforms.ToPILImage()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):

                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                # score: [n, c, h, w]
                score = self.model(data)
                pred = F.softmax(score, dim=1).max(1)[1]
                image = pred.cpu().clone().float()
                print(image.shape)

                image = unloader(image)
                # image = image.squeeze(0)
                image.save('./fig/pred_{}.png'.format(batch_idx))

        if training:
            self.model.train()
