# Author: Kien Tran (github.com/trantrikien239). 
# Reference code: 
# - https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
# - Torch inception model: https://pytorch.org/hub/pytorch_vision_inception_v3/

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random, randint
from scipy.linalg import sqrtm


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)

    # import pdb; pdb.set_trace()
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


if __name__ == "__main__":
    import torch
    from PIL import Image
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model.eval()

    # define two fake collections of images
    img_size = 256
    # batch_size = 1
    images1 = randint(0, 255, img_size*img_size*3)
    images1 = images1.reshape((img_size,img_size,3))
    images1 = Image.fromarray(images1.astype('uint8'), 'RGB')

    images2 = randint(0, 255, img_size*img_size*3)
    images2 = images2.reshape((img_size,img_size,3))
    images2 = Image.fromarray(images2.astype('uint8'), 'RGB')


    images3 = randint(0, 255, img_size*img_size*3)
    images3 = images3.reshape((img_size,img_size,3))
    images3 = Image.fromarray(images3.astype('uint8'), 'RGB')


    # convert integer to floating point values
    # images1 = torch.tensor(images1.astype('float32'))
    # images2 = torch.tensor(images2.astype('float32'))
    images1 = preprocess(images1)
    images2 = preprocess(images2)
    images3 = preprocess(images3)

    batch12 = torch.stack([images1, images2]*10)
    batch23 = torch.stack([images1, images3]*10)


    # calculate activations
    act1 = model(batch12).detach().numpy()
    act2 = model(batch23).detach().numpy()


    # fid between images1 and images1
    fid = calculate_fid(act1, act1)
    print('FID (same): %.3f' % fid)
    # fid between images1 and images2
    fid = calculate_fid(act1, act2)
    print('FID (different): %.3f' % fid)