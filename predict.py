import argparse
import os
import random
import sys

import timm
from net.resnet import *
from torchvision import transforms
from PIL import Image


def predict(model, img):
    torch.cuda.empty_cache()
    _ = model.eval()

    with torch.no_grad():
        ### For all test images, extract features
        X = model(img.unsqueeze(0).cuda())

    return X


if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # set random seed for all gpus

    parser = argparse.ArgumentParser(description=
                                     'Official implementation of `Proxy Anchor Loss for Deep Metric Learning`'
                                     + 'Our code is modified from `https://github.com/dichotomies/proxy-nca`'
                                     )

    parser.add_argument('--embedding-size', default=512, type=int,
                        dest='sz_embedding',
                        help='Size of embedding that is appended to backbone model.'
                        )
    parser.add_argument('--gpu-id', default=0, type=int,
                        help='ID of GPU that is used for training.'
                        )
    parser.add_argument('--model', default='bn_inception',
                        help='Model for training'
                        )
    parser.add_argument('--resume', default='',
                        help='Path of resuming model'
                        )
    parser.add_argument('--img-path', default='',
                        help='Path of test image'
                        )
    args = parser.parse_args()

    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)

    # Data Root Directory
    os.chdir('./data/')
    data_root = os.getcwd()

    # Backbone Model
    if args.model.find('resnet18') + 1:
        model = Resnet18(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=1)
    elif args.model.find('resnet50') + 1:
        model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=1)
    elif args.model.find('resnet101') + 1:
        model = Resnet101(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=1)
    else:
        model = timm.create_model(args.model, pretrained=False, num_classes=0)
    model = model.cuda()

    if os.path.isfile(args.resume):
        print('=> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('=> No checkpoint found at {}'.format(args.resume))
        sys.exit(0)
    resnet_sz_resize = 256
    resnet_sz_crop = 224
    resnet_mean = [0.485, 0.456, 0.406]
    resnet_std = [0.229, 0.224, 0.225]
    resnet_transform = transforms.Compose([
        transforms.Resize(resnet_sz_resize),
        transforms.CenterCrop(resnet_sz_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=resnet_mean, std=resnet_std)
    ])
    img = Image.open(args.img_path)
    img = resnet_transform(img)
    with torch.no_grad():
        print("**Evaluating...**")
        embedding = predict(model, img)
        print(embedding)
