from skimage import io as skio, color
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
import zipfile
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import io
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import log_softmax
import math
import pickle
import os
from torch.autograd import Function
USE_CUDA = torch.cuda.is_available() #helper to check if gpu is available


class ABGamut:
    RESOURCE_POINTS = 'ab-gamut.npy'
    RESOURCE_PRIOR = 'q-prior.npy'

    DTYPE = np.float32
    EXPECTED_SIZE = 313

    def __init__(self):
        self.points = np.load(self.RESOURCE_POINTS).astype(self.DTYPE)
        self.prior = np.load(self.RESOURCE_PRIOR).astype(self.DTYPE)
        assert self.points.shape == (self.EXPECTED_SIZE, 2)

class CIELAB:
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

class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels):
        softmax = log_softmax(outputs, dim=1)

        norm = labels.clone()

        norm[norm != 0] = torch.log(norm[norm != 0])

        return -torch.sum((softmax - norm) * labels) / outputs.shape[0]

class AnnealedMeanDecodeQ:
    def __init__(self, cielab, T, device='cuda'):
        self.q_to_ab = torch.from_numpy(cielab.q_to_ab).to(device)

        self.T = T

    def __call__(self, q):
        if self.T == 0:
            # makeing this a special case is somewhat ugly but I have found
            # no way to make this a special case of the branch below (in
            # NumPy that would be trivial)
            ab = self._unbin(self._mode(q))
        else:
            q = self._annealed_softmax(q)

            a = self._annealed_mean(q, 0)
            b = self._annealed_mean(q, 1)
            ab = torch.cat((a, b), dim=1)

        return ab.type(q.dtype)

    def _mode(self, q):
        return q.max(dim=1, keepdim=True)[1]

    def _unbin(self, q):
        _, _, h, w = q.shape

        ab = torch.stack([
            self.q_to_ab.index_select(
                0, q_.flatten()
            ).reshape(h, w, 2).permute(2, 0, 1)

            for q_ in q
        ])

        return ab

    def _annealed_softmax(self, q):
        q = torch.exp(q / self.T)
        q /= q.sum(dim=1, keepdim=True)

        return q

    def _annealed_mean(self, q, d):
        am = torch.tensordot(q, self.q_to_ab[:, d], dims=((1,), (0,)))

        return am.unsqueeze(dim=1)

class GetClassWeights:
    def __init__(self, cielab, lambda_=0.5, device='cuda'):
        prior = torch.from_numpy(cielab.gamut.prior).to(device)

        uniform = torch.zeros_like(prior)
        uniform[prior > 0] = 1 / (prior > 0).sum().type_as(uniform)

        self.weights = 1 / ((1 - lambda_) * prior + lambda_ * uniform)
        self.weights /= torch.sum(prior * self.weights)

    def __call__(self, ab_actual):
        return self.weights[ab_actual.argmax(dim=1, keepdim=True)]

class RebalanceLoss(Function):
    @staticmethod
    def forward(ctx, data_input, weights):
        ctx.save_for_backward(weights)

        return data_input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        weights, = ctx.saved_tensors

        # reweigh gradient pixelwise so that rare colors get a chance to
        # contribute
        grad_input = grad_output * weights

        # second return value is None since we are not interested in the
        # gradient with respect to the weights
        return grad_input, None

#CustomDataset is subclass of Dataset.
class CustomDataset(Dataset):
    def __init__(self, files, file_dir):
        # load zip dataset
        self.files = files
        self.file_dir = file_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        with open(self.file_dir+file, 'rb') as f:
            x = pickle.load(f)
        return x[1]/100.0, x[2]

class ECCVGenerator(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGenerator, self).__init__()

        model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]

        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]

        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]

        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]

        model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        conv1_2 = self.model1(input_l)
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
#         out_reg = self.model_out(conv8_3)

#         return self.upsample4(out_reg)
        return conv8_3


class train:
    def train_model(self, model, loss_fn, optimizer, train_generator, dev_generator, epochs, model_path='model.bin'):
        """
        Helper function to train a pytorch model.
        :param model: a pytorch model
        :param loss_fn: a function that can calculate loss between the predicted and gold labels
        :param optimizer: a pytorch optimizer like sgd or adam
        :param train_generator: a DataLoader that provides batches of the training set
        :param dev_generator: a DataLoader that provides batches of the dev set
        :param epochs: number of epochs to train the model
        :reshape_input: Boolean value to determine if input image has to be flattened
        :model_path: path to store model
        """

        device_type = 'cuda' if USE_CUDA else 'cpu'
        device = torch.device(device_type) #set device as cuda or cpu depending if we are training on gpu
        model.to(device) #model needs to be transferred to gpu or cpu for training

        print("Training model")

        dev_loss_max = 9999999.9999

        get_class_weights = GetClassWeights(CIELAB(), device=device_type)
        rebalance_loss = RebalanceLoss.apply

        for epoch in range(epochs):
            totalLoss = 0
            model.train() #set the model in training model, lets the layers know that they are training mode
            for batch, labels in tqdm(train_generator, position=0, leave=True):
                batch = batch.to(device) #transfer the input feature to gpu or cpu
                labels = labels.to(device) #transfer the labels to gpu or cpu
                optimizer.zero_grad() #Set all graidents to zero for each step as they accumulate over backprop

                outputs = model(batch) #single forward pass of the model
                color_weights = get_class_weights(labels.double())
                outputs = rebalance_loss(outputs.double(), color_weights)

                loss = loss_fn(outputs.double(), labels.double()) #computes the loss with gold labels

                totalLoss+=loss
                loss.backward() #loss.backward() computes dloss/dx for every parameter x which has requires_grad=True
                optimizer.step() #x += -lr * x.grad ie updates the weights of the parameters
            print("On epoch", epoch)
            print("Traning Loss:", totalLoss.data)
            self.save_model(model, model_path)
    #         dev_loss = test_model(model, loss_fn, dev_generator, False, reshape_input=reshape_input)
    #         if dev_loss < dev_loss_max:
    #             save_model(model, model_path)
        return model

    def get_preds(self, model, image):
        """
        Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
        :param model: a pytorch model
        :param image: a black and white image 1xhxw
        """

        device = torch.device('cuda' if USE_CUDA else 'cpu')
        model.to(device)

        model.eval() #sets the model in evaluation mode

        # Iterate over batches in the test dataset
        with torch.no_grad(): #do not compute the gradienets of the model
            h = image.shape[1]
            w = image.shape[2]
            img = image.reshape(1,1,h,w)
            img = torch.tensor(img).to(device)
            preds = model(img)
        return preds.cpu().detach().numpy()

    def save_model(self, model, file='model.bin'):
        torch.save(model.state_dict(), file) #state_dict is dictionary object that maps each layer to its parameter tensor, hence we save that

    def load_model(self, model, file='model.bin'):
        device = torch.device('cuda' if USE_CUDA else 'cpu')
        model.load_state_dict(torch.load(file, map_location=device)) #load the model's state dict from the saved file
        return model

    def train_model_with_args(self, images_path, epochs, loss_function, batch_size, model_path):
        """
        Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
        :param images_path: folder path to preprocessed lab images
        :param epochs: maximum number of epochs to run the model
        :param loss_function: mse or cross_entropy
        :batch_size: training batch size
        :model_path: path to store the trained model
        """
        X = os.listdir(images_path)
        # divide dataset into train, val and test sets
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        X_val, X_test = train_test_split(X_test, test_size=0.2, random_state=42)

        train_dataset = CustomDataset(X_train, images_path)
        val_dataset = CustomDataset(X_val, images_path)
        test_dataset = CustomDataset(X_test, images_path)

        train_loader = DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True,
            pin_memory=True)

        val_loader = DataLoader(
            val_dataset,
            batch_size = batch_size,
            shuffle = True,
            pin_memory=True)

        test_loader = DataLoader(
            test_dataset,
            batch_size = batch_size,
            shuffle = True,
            pin_memory=True)

        model = ECCVGenerator()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        if loss_function == "mse":
            loss_fn = nn.MSELoss()
        else:
            loss_fn = CrossEntropyLoss2d()
        model_trained = self.train_model(model, loss_fn, optimizer, train_loader, val_loader, epochs=epochs)
        self.save_model(model_trained, model_path)

    def convert_image_to_color(self, raw_image_path, store_image_path, model_path):
        data_transforms_test = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop((224,224)),
                    transforms.ToTensor()
                ])
        model = ECCVGenerator()
        model_trained = self.load_model(model, model_path)
        
        print("Loaded model!")
        img = Image.open(raw_image_path) #read image as PIL object
        img = data_transforms_test(img) #apply transformation to image
        img = color.rgb2lab(img.permute(1,2,0))
        img = img.transpose(2,0,1)
        l, ab = img[:1,:,:], img[1:, :,:]

        ab_preds = self.get_preds(model_trained, l/100.0)[0, :, :, :]
        cielab = CIELAB()
        ab_new = torch.tensor([ab_preds])
        decode_q = AnnealedMeanDecodeQ(cielab,T=0.38,device='cpu')
        ab_new = decode_q(ab_new)
        ab_ = F.interpolate(torch.tensor(ab_new), (224,224), mode='bilinear')
        ab_ = ab_[0,:,:,:]
        t = np.concatenate((l,ab_)).transpose(1,2,0)
        
        print("Generated colorized image")
        plt.imshow(color.lab2rgb(t))
        plt.imsave(store_image_path, color.lab2rgb(t))
        print("Image succesfully colorized and saved!")
