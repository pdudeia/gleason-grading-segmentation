import os
from glob import glob
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
import torch
import torchvision.transforms.functional as TF
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.style as style
from torch.utils.tensorboard import SummaryWriter
import time


class Prostate_data(Dataset):

    def __init__(self, img_path='../harvard_data/TMA_Images', mask_path='../harvard_data/Gleason_masks_train',
                 dataset_type='train', img_size=3100, valid_split=['ZT76'], test_split=['ZT80'], num_classes=5):
        self.img_path = img_path
        # ground_truths
        self.mask_path = mask_path
        self.img_size = img_size
        self.num_classes = num_classes

        self.file_names = []
        # dataset_type : train, valid, test
        self.dataset_type = dataset_type
        # valid_split, test_split : keywords to identify valid & test data
        slide_dict = {'valid': valid_split, 'test': test_split}
        # flag_dict : dictionary with keys as file names, used in getitem for book-keeping in dynamic pre-processing
        self.flag_dict = {}

        for file in glob(self.img_path + '/*.jpg'):
            # file : full path, _file_name extracts the image name
            _file_name = file.split('\\')[-1]
            # _file_name form is ZTxx_*.jpg, ZTxx decides the dataset type
            _slide_type = _file_name.split('.')[0].split('_')[0]
            if dataset_type == 'train':
                if not (_slide_type in valid_split) and not (_slide_type in test_split):
                    for fname in self.all_files(_file_name):
                        self.file_names.append(fname)
                        self.flag_dict[fname] = False
            else:
                if _slide_type in slide_dict[dataset_type]:
                    self.file_names.append(_file_name)
                    self.flag_dict[_file_name] = False
        random.seed(10)
        random.shuffle(self.file_names)
        # setup data, transform to be used in getitem (standard ImageNet normalization since we use pre-trained ResNet)
        self.data = {}
        self.transform = {
            'train': transforms.Compose([transforms.ColorJitter(0.2,0.2,0.2,0.2),transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            'valid': transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }

    def __len__(self):
        return len(self.file_names)

    def all_files(self,_file_name):
        # helper function for training data transformations
        # return modified file names with appended transformation types
        return [_file_name,_file_name+'_tranhor',_file_name+'_tranver']

    def __getitem__(self, idx):
        _file_name = self.file_names[idx]
        _file_flag = self.flag_dict[_file_name]

        # _file_flag tells whether the preprocessing has already been done
        if _file_flag:
            return self.data[_file_name]
        else:
            # hard coded : if transforms are applied, _tran will be added to file name, remove that to get image path
            img_path = self.img_path+'/'+_file_name.split('_tran')[0] if '_tran' in _file_name else self.img_path + '/' + _file_name
            mask_path = self.mask_path + '/' + 'mask_' + _file_name.split('_tran')[0].split('.')[0] + '.png' if '_tran' in _file_name else self.mask_path + '/' + 'mask_' + _file_name.split('.')[0] + '.png'

            # load image and ground truth
            img = Image.open(img_path).resize((self.img_size, self.img_size)).convert('RGB')
            mask = Image.open(mask_path).resize((self.img_size, self.img_size)).convert('RGB')
            # apply transforms
            if 'hor' in _file_name:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if 'ver' in _file_name:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            if 'aff' in _file_name:
                img = TF.affine(img,20,(0,0),1.35,0)
                mask = TF.affine(mask,20,(0,0),1.35,0)

            # create one-hot mask
            mask_array = np.asarray(mask)
            # oneh_mask is the one-hot ground truth for segmentation, size = (num_classes,h,w)
            oneh_mask = np.zeros((self.num_classes, self.img_size, self.img_size))
            for x in range(self.img_size):
                for y in range(self.img_size):
                    pixel_class = self.get_class(mask_array[x, y,:])
                    oneh_mask[pixel_class, x, y] = 1

            # since a large portion of many images is white, for better soft dice values we the rgb values close to 0
            array_img = np.asarray(img)
            # temporary copy of the image
            timg = copy.deepcopy(array_img)
            for x in range(self.img_size):
                for y in range(self.img_size):
                    rgb_n = array_img[x, y, :] / 255.0
                    if rgb_n[0] > 0.8 and rgb_n[1] > 0.8 and rgb_n[2] > 0.8:
                        timg[x, y, :] = [0, 0, 0]
            final_img = Image.fromarray(timg.astype('uint8'), 'RGB')

            # final training data
            img_tensor = self.transform[self.dataset_type](final_img)
            mask_tensor = torch.from_numpy(oneh_mask).view(self.num_classes, self.img_size, self.img_size)
            # save preprocessed image and update flag dictionary
            self.data[_file_name] = (img_tensor, mask_tensor)
            self.flag_dict[_file_name] = True
            return self.data[_file_name]

    def get_class(self, rgb):
        '''
        takes in rgb values of the pixel and returns the class of the pixel
        used thresholding since annotated values may not be all equal
        '''
        rgb_n = rgb / 255.0

        # white
        if rgb_n[0] > 0.8 and rgb_n[1] > 0.8 and rgb_n[2] > 0.8:
            return 4
        # red
        elif rgb_n[0] > 0.8 and rgb_n[1] < 0.8 and rgb_n[2] < 0.8:
            return 3
        # yellow
        elif rgb_n[0] > 0.8 and rgb_n[1] > 0.8 and rgb_n[2] < 0.8:
            return 2
        # green
        elif rgb_n[0] < 0.8 and rgb_n[1] > 0.8 and rgb_n[2] < 0.8:
            return 0
        # blue
        elif rgb_n[0] < 0.8 and rgb_n[1] < 0.8 and rgb_n[2] > 0.8:
            return 1
        else:
            print(rgb_n)
            raise ValueError('Weird rgb combination! Did not match any of 5 classes.')

def soft_dice_loss(y_pred,y_true):
    '''
    y_pred: (-1,5,512,512) : predictions
    y_true: (-1,512,512,5) : targets
    compute the soft dice loss : 2 |AB| / (|A|^2 + |B|^2)

    '''
    y_true = y_true.view(-1,5,512,512)
    epsilon = 1e-7
    dice_numerator = epsilon + 2 * torch.sum(y_true*y_pred,axis=(2,3))
    dice_denominator = epsilon + torch.sum(y_true*y_true,axis=(2,3)) + torch.sum(y_pred*y_pred,axis=(2,3))
    dice_loss = 1 - torch.mean(dice_numerator/dice_denominator)

    return dice_loss

class FocalDiceloss(nn.Module):
# Calculate a combination of focal and dice loss

    def __init__(self,gamma=0):
        super(FocalDiceloss,self).__init__()
        self.gamma = gamma

    def forward(self,outputs,targets_oneh,targets):
        # Focal loss : (1-p_t)^gamma log(p_t), where p_t is true label prediction probability (label could be 0 or 1)
        # softmax outputs
        soft_outs = torch.nn.functional.softmax(outputs,dim=1)
        # logits
        log_soft = torch.nn.functional.log_softmax(outputs,dim=1)
        # weigh logits for focal loss
        weight_loss = torch.pow((1 - soft_outs),self.gamma) * log_soft
        loss = 0.4*torch.nn.functional.nll_loss(weight_loss,targets) + 0.6*soft_dice_loss(outputs,targets_oneh)
        return loss

def main():
    from learner import Learner
    from res_unet_dropout import ResUnet
    # dropout probability
    dprob=0.2
    epochs = 8

    trainset = Prostate_data(img_size=512, num_classes=5)
    validset = Prostate_data(dataset_type='valid', img_size=512, num_classes=5)
    datasets = {'train': trainset, 'valid': validset}

    for lr in [1e-4,5e-4,1e-3,5e-3,1e-2]:
        for gamma in [0,1,2] :

            model = ResUnet(num_classes=5,dprob=dprob)
            criterion = FocalDiceloss(gamma=gamma)
            optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10)
            dtime = '0057_1806'
            # tb_logs : for tensorboard
            tb_logs = {'path':'logdirs/trials_aug/respre_SGD_plateau','comment':f'lr={lr}_gamma={gamma}_dprob={dprob}_{dtime}'}
            trainer = Learner(datasets,model,criterion,optimizer,scheduler,bs=8,num_workers=4)
            try :
                trainer.fit(tb_logs=tb_logs,epochs=epochs)
                # torch.save(trainer.model,f'logdirs/onevall_trials_aug/respre_SGD_plateau/lr={lr}_gamma={gamma}_dprob={dprob}_{dtime}/{dtime}')
            except KeyboardInterrupt:
                pass
                # torch.save(trainer.model,f'logdirs/onevall_trials_aug/respre_SGD_plateau/lr={lr}_gamma={gamma}_dprob={dprob}_{dtime}/{dtime}')


if __name__=='__main__':
    main()

# lr = 5e-4
# gamma = 1
# dtime = '0057_1806'
# testset = Prostate_data(dataset_type='test',img_size=512,num_classes=5)
# model = torch.load(f'logdirs/onevall_trials_aug/respre_SGD_plateau/lr={lr}_gamma={gamma}_dprob={dprob}_{dtime}/{dtime}')
# model.eval()
# test_dataloader = DataLoader(testset,batch_size=4)
# kappa = 0
# for inputs, targets in tqdm(test_dataloader):
#     inputs = inputs.float().view(-1, 3, 512, 512).to(torch.device('cuda:0'))
#     targets_oneh = targets.view(-1, model.num_classes, 512, 512).to(torch.device('cuda:0'))
#     targets = torch.argmax(targets_oneh,dim=1)
#     outputs = model(inputs)
#     preds = torch.argmax(outputs,dim=1)
#     kappa += Kappa(preds,targets)
# print(f'Kappa score = {kappa/len(testset)}')
