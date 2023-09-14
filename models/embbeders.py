import torch
import clip
import torchvision.transforms.functional as F

from torchvision import models as torch_models

from .arcface import Backbone


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_SIZE = 224
ATTRIBUTE_SIZE = 224
ARCFACE_SIZE = 150
ARCFACE_CROP = 112


class IdentEmbedder(torch.nn.Module):
    def __init__(self, extract_keys=['arcface', 'dense']):
        super().__init__()
        self.extract_keys = extract_keys

    def forward(self, batch):
        xs = list()
        for key in self.extract_keys:
            batch_size = batch[key].size(0)
            x = batch[key].view(batch_size, -1)
            xs.append(x)

        xs = torch.cat(xs, dim=1)
        return xs.unsqueeze(dim=1)


class DINOEmbedder(torch.nn.Module):
    def __init__(self, extract_keys=['image']):
        super(DINOEmbedder, self).__init__()
        self.extract_keys = extract_keys
        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.dino.head = torch.nn.Identity()
        self.dino.eval()
        for module in [self.dino]:
            for param in module.parameters():
                param.requires_grad = False

        self.register_buffer('mean', torch.Tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def _preprocess(self, x):
        # x in [-1 1]
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        return x

    def forward(self, batch):
        xs = list()
        for key in self.extract_keys:
            x = self.dino(self._preprocess(batch[key]))
            x = x.flatten(start_dim=1)
            xs.append(x)

        xs = torch.cat(xs, dim=1)
        return xs.unsqueeze(dim=1)


class CLIPImageEmbedder(torch.nn.Module):
    def __init__(self, extract_keys=['image'], model_name='ViT-L/14'):
        super(CLIPImageEmbedder, self).__init__()
        self.extract_keys = extract_keys
        self.clip, _ = clip.load(model_name)
        self.clip.eval()
        for module in [self.clip]:
            for param in module.parameters():
                param.requires_grad = False

        self.register_buffer('mean', torch.Tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def _preprocess(self, x):
        # x in [-1 1]
        x = F.center_crop(F.resize(x, CLIP_SIZE), CLIP_SIZE)
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        return x

    def forward(self, batch):
        xs = list()
        for key in self.extract_keys:
            x = batch[key]
            x = self.clip.encode_image(self._preprocess(x))
            x = x.flatten(start_dim=1)
            xs.append(x)

        xs = torch.cat(xs, dim=1)
        return xs.unsqueeze(dim=1)


class TransferModel(torch.nn.Module):
    def __init__(self, backbone, num_classes):
        super(TransferModel, self).__init__()
        self.backbone = backbone

        num_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return features


class AttributeEmbedder(torch.nn.Module):
    def __init__(
        self,
        path='./pretrained/attr-classifier-resnet50-gs/classifier_last.pt',
        extract_keys=['image'],
    ):
        super(AttributeEmbedder, self).__init__()
        self.extract_keys = extract_keys
        self.attr = TransferModel(backbone=torch_models.resnet50(pretrained=True), num_classes=40)
        self.attr.load_state_dict(torch.load(path))
        self.attr.eval()
        for module in [self.attr]:
            for param in module.parameters():
                param.requires_grad = False

        self.register_buffer('mean', torch.Tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def _preprocess(self, x):
        # x in [-1 1]
        x = F.center_crop(F.resize(x, ATTRIBUTE_SIZE), ATTRIBUTE_SIZE)
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        return x

    def forward(self, batch):
        xs = list()
        for key in self.extract_keys:
            feat = self.attr(self._preprocess(batch[key]))
            x = torch.sigmoid(feat).flatten(start_dim=1)
            xs.append(x)

        xs = torch.cat(xs, dim=1)
        return xs.unsqueeze(dim=1)


class ArcFaceEmbedder(torch.nn.Module):
    def __init__(self, path='./pretrained/arcface/model_ir_se50.pth', extract_keys=['image']):
        super(ArcFaceEmbedder, self).__init__()
        self.extract_keys = extract_keys
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(path))
        self.facenet.eval()
        for module in [self.facenet, self.face_pool]:
            for param in module.parameters():
                param.requires_grad = False

    def _preprocess(self, x):
        # x in [-1 1]
        x = F.center_crop(F.resize(x, ARCFACE_SIZE), ARCFACE_CROP)
        return x

    def forward(self, batch):
        xs = list()
        for key in self.extract_keys:
            x = self.facenet(self._preprocess(batch[key]))
            xs.append(x)

        xs = torch.cat(xs, dim=1)
        return xs.unsqueeze(dim=1)


class ClipImageIdentEmbedder(torch.nn.Module):
    def __init__(self, extract_keys=['arcface', 'dense']):
        super().__init__()
        self.clipper = CLIPImageEmbedder()
        self.extract_keys = extract_keys

    def forward(self, batch):
        xs = list()
        for key in self.extract_keys:
            batch_size = batch[key].size(0)
            x = batch[key].view(batch_size, -1)
            xs.append(x)

        xs.append(self.clipper(batch).view(batch_size, -1))
        xs = torch.cat(xs, dim=1)
        return xs.unsqueeze(dim=1)