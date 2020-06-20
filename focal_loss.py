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
        self.mask_path = mask_path
        self.img_size = img_size
        self.num_classes = num_classes
        self.file_names = []
        self.dataset_type = dataset_type
        slide_dict = {'valid': valid_split, 'test': test_split}
        self.flag_dict = {}
        for file in glob(self.img_path + '/*.jpg'):
            _file_name = file.split('\\')[-1]
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
        return [_file_name,_file_name+'_tranhor',_file_name+'_tranver']

    def __getitem__(self, idx):
        _file_name = self.file_names[idx]
        _file_flag = self.flag_dict[_file_name]
        if _file_flag:
            return self.data[_file_name]
        else:
            img_path = self.img_path+'/'+_file_name.split('_tran')[0] if '_tran' in _file_name else self.img_path + '/' + _file_name
            mask_path = self.mask_path + '/' + 'mask_' + _file_name.split('_tran')[0].split('.')[0] + '.png' if '_tran' in _file_name else self.mask_path + '/' + 'mask_' + _file_name.split('.')[0] + '.png'

            img = Image.open(img_path).resize((self.img_size, self.img_size)).convert('RGB')
            mask = Image.open(mask_path).resize((self.img_size, self.img_size)).convert('RGB')
            ## transforms
            if 'hor' in _file_name:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if 'ver' in _file_name:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            if 'aff' in _file_name:
                img = TF.affine(img,20,(0,0),1.35,0)
                mask = TF.affine(mask,20,(0,0),1.35,0)

            mask_array = np.asarray(mask)
            oneh_mask = np.zeros((self.num_classes, self.img_size, self.img_size))
            for x in range(self.img_size):
                for y in range(self.img_size):
                    pixel_class = self.get_class(mask_array[x, y,:])
                    oneh_mask[pixel_class, x, y] = 1

            array_img = np.asarray(img)
            timg = copy.deepcopy(array_img)
            for x in range(self.img_size):
                for y in range(self.img_size):
                    rgb_n = array_img[x, y, :] / 255.0
                    if rgb_n[0] > 0.8 and rgb_n[1] > 0.8 and rgb_n[2] > 0.8:
                        timg[x, y, :] = [0, 0, 0]
            final_img = Image.fromarray(timg.astype('uint8'), 'RGB')

            img_tensor = self.transform[self.dataset_type](final_img)
            mask_tensor = torch.from_numpy(oneh_mask).view(self.num_classes, self.img_size, self.img_size)
            self.data[_file_name] = (img_tensor, mask_tensor)
            self.flag_dict[_file_name] = True
            return self.data[_file_name]

    def get_class(self, rgb):
        '''
        takes in rgb values of the pixel and returns the class of the pixel
        '''
        rgb_n = rgb / 255.0

        # white
        if rgb_n[0] > 0.8 and rgb_n[1] > 0.8 and rgb_n[2] > 0.8:
            return 2
        # red
        elif rgb_n[0] > 0.8 and rgb_n[1] < 0.8 and rgb_n[2] < 0.8:
            return 1
        # yellow
        elif rgb_n[0] > 0.8 and rgb_n[1] > 0.8 and rgb_n[2] < 0.8:
            return 1
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
    '''y_pred: (-1,5,512,512) :predictions
       y_true: (512,512,5) : targets
       compute the soft dice loss

       '''
    y_true = y_true.view(-1,3,256,256)
    epsilon = 1e-7
    dice_numerator = epsilon + 2 * torch.sum(y_true*y_pred,axis=(2,3))
    dice_denominator = epsilon + torch.sum(y_true*y_true,axis=(2,3)) + torch.sum(y_pred*y_pred,axis=(2,3))
    dice_loss = 1 - torch.mean(dice_numerator/dice_denominator)

    return dice_loss


def show_train_predictions(model,trainset,device,idx_list):
    fig,axes = plt.subplots(nrows=len(idx_list),ncols=3,figsize=(15,15))
    for i in range(len(idx_list)):
        idx = idx_list[i]
        input_img = Image.fromarray(np.asarray(trainset[idx][0].view(trainset.img_size,trainset.img_size,3).squeeze()).astype('uint8'), 'RGB')
        target_img = get_rgb(trainset[idx][1].squeeze())
        with torch.no_grad():
            pred_img = get_rgb(model(trainset[idx][0].view(-1,3,trainset.img_size,trainset.img_size).float().to(device)).squeeze())
        if len(idx_list)>1:
            axes[i,0].imshow(input_img)
            axes[i,1].imshow(pred_img)
            axes[i,2].imshow(target_img)
        else:
            axes[0].imshow(input_img)
            axes[1].imshow(pred_img)
            axes[2].imshow(target_img)
    if len(idx_list)>1:
        axes[0,0].set_title('T Input')
        axes[0,1].set_title('T Prediction')
        axes[0,2].set_title('T Target')
    else:
        axes[0].set_title('T Input')
        axes[1].set_title('T Prediction')
        axes[2].set_title('T Target')

    return fig

def show_valid_predictions(model,validset,device,idx_list):
    fig,axes = plt.subplots(nrows=len(idx_list),ncols=3,figsize=(15,15))
    for i in range(len(idx_list)):
        idx = idx_list[i]
        input_img = Image.fromarray(np.asarray(validset[idx][0].view(validset.img_size,validset.img_size,3).squeeze()).astype('uint8'), 'RGB')
        target_img = get_rgb(validset[idx][1].squeeze())
        with torch.no_grad():
            pred_img = get_rgb(model(validset[idx][0].view(-1,3,validset.img_size,validset.img_size).float().to(device)).squeeze())
        if len(idx_list)>1:
            axes[i,0].imshow(input_img)
            axes[i,1].imshow(pred_img)
            axes[i,2].imshow(target_img)
        else:
            axes[0].imshow(input_img)
            axes[1].imshow(pred_img)
            axes[2].imshow(target_img)
    if len(idx_list)>1:
        axes[0,0].set_title('V Input')
        axes[0,1].set_title('V Prediction')
        axes[0,2].set_title('V Target')
    else:
        axes[0].set_title('V Input')
        axes[1].set_title('V Prediction')
        axes[2].set_title('V Target')

    return fig

def get_rgb(tensor_img):
    pallete_dict = {
        0 : [0,255,0],
        1 : [0,0,255],
        2 : [255,255,255],
        3 : [255,0,0],
        4 : [255,255,0]
    }
    img_h = tensor_img.size()[2]
    out_img = np.zeros((img_h,img_h,3))
    for h in range(img_h):
        for w in range(img_h):
            pixel_class = torch.argmax(tensor_img[:,h,w]).item()
            out_img[h,w,:] = pallete_dict[pixel_class]
    final_img = Image.fromarray(out_img.astype('uint8'), 'RGB')
    return final_img

class Focalloss(nn.Module):

    def __init__(self,gamma=0):
        super(Focalloss,self).__init__()
        self.gamma = gamma

    def forward(self,outputs,targets_oneh,targets):
        soft_outs = F.softmax(outputs,dim=1)
        log_soft = F.log_softmax(outputs,dim=1)
        weight_loss = torch.pow((1 - soft_outs),self.gamma) * log_soft
        loss = 0.4*F.nll_loss(weight_loss,targets) + 0.6*soft_dice_loss(outputs,targets_oneh)
        return loss

def main():
    from learner import Learner
    from res_unet_dropout import ResUnet
    # lr=3e-5
    dprob=0.2
    epochs = 8

    trainset = Prostate_data(img_size=256, num_classes=3)
    validset = Prostate_data(dataset_type='valid', img_size=256, num_classes=3)
    datasets = {'train': trainset, 'valid': validset}

    for lr in [1e-4,5e-4,1e-3,5e-3,1e-2]:
        for gamma in [0] :
            # fig,axes = plt.subplots(nrows=1,ncols=6,figsize=(24,4))
            # imgs = []
            # for i in tqdm(range(6)):
            #     imgs.append(get_rgb(trainset[i][1]))
            # for j in range(6):
            #     axes[j].imshow(imgs[j])
            # plt.show()

            model = ResUnet(num_classes=3,dprob=dprob)
            criterion = Focalloss(gamma=gamma)
            optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10)
            dtime = '0057_1806'
            tb_logs = {'path':'logdirs/onevall_trials_aug/respre_SGD_plateau','comment':f'lr={lr}_gamma={gamma}_dprob={dprob}_{dtime}'}
            trainer = Learner(datasets,model,criterion,optimizer,scheduler,bs=8,num_workers=4)
            try :
                trainer.fit(tb_logs=tb_logs,epochs=epochs)
                # torch.save(trainer.model,f'logdirs/onevall_trials_aug/respre_SGD_plateau/lr={lr}_gamma={gamma}_dprob={dprob}_{dtime}/{dtime}')
            except KeyboardInterrupt:
                pass
                # torch.save(trainer.model,f'logdirs/onevall_trials_aug/respre_SGD_plateau/lr={lr}_gamma={gamma}_dprob={dprob}_{dtime}/{dtime}')


if __name__=='__main__':
    main()