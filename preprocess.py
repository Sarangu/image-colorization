import zipfile
import io
from PIL import Image
from skimage import io as skio, color
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
import math
import numpy as np
import torch.nn as nn
import pickle
import json
import base64
from tqdm import tqdm
import os

class ABGamut:
    RESOURCE_POINTS = 'ab-gamut.npy'

    DTYPE = np.float32
    EXPECTED_SIZE = 313

    def __init__(self):
        self.points = np.load(self.RESOURCE_POINTS).astype(self.DTYPE)
        assert self.points.shape == (self.EXPECTED_SIZE, 2)


class CIELAB:
    L_MEAN = 50

    AB_BINSIZE = 10
    AB_RANGE = [-125 - AB_BINSIZE // 2, 125 + AB_BINSIZE // 2, AB_BINSIZE]
    AB_DTYPE = np.float32

    Q_DTYPE = np.int64

    RGB_RESOLUTION = 101
    RGB_RANGE = [0, 1, RGB_RESOLUTION]
    RGB_DTYPE = np.float64

    def __init__(self, gamut=None):
        self.gamut = gamut if gamut is not None else ABGamut()

        a, b, self.ab = self._get_ab()
        
        self.ab_gamut_mask = self._get_ab_gamut_mask(
            a, b, self.ab, self.gamut)

        self.ab_to_q = self._get_ab_to_q(self.ab_gamut_mask)

        self.q_to_ab = self._get_q_to_ab(self.ab, self.ab_gamut_mask)

    @classmethod
    def _get_ab(cls):
        a = np.arange(*cls.AB_RANGE, dtype=cls.AB_DTYPE)
        b = np.arange(*cls.AB_RANGE, dtype=cls.AB_DTYPE)
        
        b_, a_ = np.meshgrid(a, b)
        ab = np.dstack((a_, b_))

        return a, b, ab

    @classmethod
    def _get_ab_gamut_mask(cls, a, b, ab, gamut):
        ab_gamut_mask = np.full(ab.shape[:-1], False, dtype=bool)

        a = np.digitize(gamut.points[:, 0], a) - 1
        b = np.digitize(gamut.points[:, 1], b) - 1

        for a_, b_ in zip(a, b):
            ab_gamut_mask[a_, b_] = True

        return ab_gamut_mask

    @classmethod
    def _get_ab_to_q(cls, ab_gamut_mask):
        ab_to_q = np.full(ab_gamut_mask.shape, -1, dtype=cls.Q_DTYPE)
        ab_to_q[ab_gamut_mask] = np.arange(np.count_nonzero(ab_gamut_mask))
        return ab_to_q

    @classmethod
    def _get_q_to_ab(cls, ab, ab_gamut_mask):
        return ab[ab_gamut_mask] + cls.AB_BINSIZE / 2

    def bin_ab(self, ab):
        ab_discrete = ((ab + 110) / self.AB_RANGE[2]).astype(int)

        a, b = np.hsplit(ab_discrete.reshape(-1, 2), 2)

        return self.ab_to_q[a, b].reshape(*ab.shape[:2])

class SoftEncodeAB:
    def __init__(self, cielab, neighbours=5, sigma=5.0, device='cpu'):
        self.cielab = cielab
        self.q_to_ab = torch.from_numpy(self.cielab.q_to_ab).to(device)

        self.neighbours = neighbours
        self.sigma = sigma

    def __call__(self, ab):
        n, _, h, w = ab.shape

        m = n * h * w

        # find nearest neighbours
        ab_ = ab.permute(1, 0, 2, 3).reshape(2, -1)
        q_to_ab = self.q_to_ab.type(ab_.dtype)

        cdist = torch.cdist(q_to_ab, ab_.t())

        nns = cdist.argsort(dim=0)[:self.neighbours, :]

        # gaussian weighting
        nn_gauss = ab.new_zeros(self.neighbours, m)

        for i in range(self.neighbours):
            nn_gauss[i, :] = self._gauss_eval(
                q_to_ab[nns[i, :], :].t(), ab_, self.sigma)

        nn_gauss /= nn_gauss.sum(dim=0, keepdim=True)

        # expand
        bins = self.cielab.gamut.EXPECTED_SIZE
        q = ab.new_zeros(bins, m)

        q[nns, torch.arange(m).repeat(self.neighbours, 1)] = nn_gauss

        return q.reshape(bins, n, h, w).permute(1, 0, 2, 3)

    @staticmethod
    def _gauss_eval(x, mu, sigma):
        norm = 1 / (2 * math.pi * sigma)

        return norm * torch.exp(-torch.sum((x - mu)**2, dim=0) / (2 * sigma**2))

class preprocess():
    def __init__(self, data_dir, preprocessed_folder):
        zf = zipfile.ZipFile(data_dir)
        X = zf.namelist()[1:]

        data_transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop((224,224)),
                    transforms.ToTensor(),
                ])
        cielab = CIELAB()
        soft_encode = SoftEncodeAB(cielab)
        i = 0
        for file in tqdm(X):
            try:
                imgfile = zf.read(file) #read file from zip directly without extracting
                dataEnc = io.BytesIO(imgfile) #convert file to bytes
                img = Image.open(dataEnc) #read image as PIL object
                img = data_transforms(img) #apply transformation to image
                img = color.rgb2lab(img.permute(1,2,0))
                img = img.transpose(2,0,1)
                l, ab = img[:1,:,:], img[1:, :,:]
                ab_ = F.interpolate(torch.tensor([ab]), (56,56), mode='bilinear')
                res = soft_encode(torch.tensor(ab_))
                ab = res[0,:,:,:]
                ab = ab.cpu().detach().numpy()
                data = [file, l, ab]
                with open(preprocessed_folder+ '_' + str(i) + '.pkl', 'wb') as outfile:
                    pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
                i+=1
                #store l and ab somewhere
            except:
                pass
