import torchvision
import torch._storage_docs


def VGG16(pretrained=False):
    model = torchvision.models.vgg16(pretrained=False)
    if not pretrained:
        return model
    state_dict = torch.load('./model/vgg16.pth')
    model.load_state_dict(state_dict)
    return model
