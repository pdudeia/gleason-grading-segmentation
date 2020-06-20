from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import random
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import numpy as np
import copy
import torch.nn as nn
import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.style as style
from torch.utils.tensorboard import SummaryWriter
from res_unet_dropout import ResUnet

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
            img_tensor = self.transform[self.dataset_type](img)
            mask_tensor = torch.from_numpy(oneh_mask).view(5, self.img_size, self.img_size)
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
    '''y_pred: (-1,5,512,512) :predictions
       y_true: (512,512,5) : targets
       compute the soft dice loss

       '''
    y_true = y_true.view(-1,5,256,256)
    epsilon = 1e-7
    dice_numerator = epsilon + 2 * torch.sum(y_true*y_pred,axis=(2,3))
    dice_denominator = epsilon + torch.sum(y_true*y_true,axis=(2,3)) + torch.sum(y_pred*y_pred,axis=(2,3))
    dice_loss = 1 - torch.mean(dice_numerator/dice_denominator)

    return dice_loss


def show_train_predictions(model,trainset,device,idx_list):
    fig,axes = plt.subplots(nrows=len(idx_list),ncols=3,figsize=(20,20))
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
    fig,axes = plt.subplots(nrows=len(idx_list),ncols=3,figsize=(20,20))
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
        2 : [255,255,0],
        3 : [255,0,0],
        4 : [255,255,255]
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

class Curriculum_Learner():

    def __init__(self,datasets,model,criterion,optimizer,tb_logs=None,scheduler=None,bs=8,num_workers=6,device='cuda:0',scoring_model=None):
        self.ogdatasets = datasets
        self.datasets = copy.deepcopy(self.ogdatasets)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.scoring_model = self.model if (scoring_model is None) else scoring_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.tb_logs = tb_logs
        self.scheduler = scheduler
        self.ogdataloaders = {}
        self.ogdataset_sizes = {}
        for key in self.ogdatasets:
            self.ogdataloaders[key] = DataLoader(self.ogdatasets[key],batch_size=bs,num_workers=num_workers)
            self.ogdataset_sizes[key] = len(self.ogdatasets[key])
        self.dataloaders = copy.deepcopy(self.ogdataloaders)
        self.dataset_sizes = copy.deepcopy(self.ogdataset_sizes)
        start = time.time()
        print(f'initialising')
        for i in tqdm(range(len(self.ogdatasets['train']))):
            _ = self.ogdatasets['train'][i]
        print(f'done in {int(time.time()-start)}')


    def fit(self,epochs):
        '''
        :param tb_logs: a dictionary containing tensorboard log flag, path and comment
        :param epochs:
        :return:
        '''
        imgsize = self.datasets['train'].img_size
        self.model.to(self.device)
        self.record_dict = {'train': {'loss': [], 'acc': []}, 'valid': {'loss': [], 'acc': []}}
        for epoch in epochs:
            print(f'EPOCH : {epoch + 1}')
            for phase in self.dataloaders.keys():
                since = time.time()
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                running_loss = 0.
                running_acc = 0.
                running_hits = 0
                for inputs, targets in tqdm(self.dataloaders[phase]):
                    inputs = inputs.float().view(-1, 3, imgsize, imgsize).to(self.device)
                    targets_oneh = targets.view(-1, self.model.num_classes, imgsize, imgsize).to(self.device)
                    targets = torch.argmax(targets_oneh, dim=1)
                    with torch.set_grad_enabled(phase=='train'):
                        self.optimizer.zero_grad()
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs,dim=1)
                        loss = self.criterion(outputs, targets_oneh, targets)
                        acc = torch.mean((targets==preds).to(float)).item()
                        hits = torch.sum((preds==targets)).item()
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                        running_loss += loss.item() * inputs.size()[0]
                        running_acc += acc*inputs.size()[0]
                        running_hits += hits
                #                 if phase == 'train' and not(scheduler is None):
                #                     scheduler.step(epoch_loss)

                epoch_loss = running_loss/self.dataset_sizes[phase]
                epoch_acc = running_acc/self.dataset_sizes[phase]

                if phase == 'valid' and not (self.scheduler is None):  # FOR SINGLE SAMPLE
                    self.scheduler.step(epoch_loss)

                if self.tb_logs is not None:
                    if phase == 'train':
                        self.tb.add_scalar('Train Loss', epoch_loss, epoch)
                        self.tb.add_scalar('Train Acc',epoch_acc,epoch)
                    else:
                        self.tb.add_scalar('Valid Loss', epoch_loss, epoch)
                        self.tb.add_scalar('Valid Acc', epoch_acc, epoch)

                self.record_dict[phase]['loss'].append(epoch_loss)
                self.record_dict[phase]['acc'].append(epoch_acc)

                print('{} Loss: {:.4f} Acc: {:.4f} Hits: {}'.format(phase, epoch_loss,epoch_acc,running_hits))
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                # deep copy the model
                if phase == 'valid' and epoch_loss < self.best_valid_loss:
                    if len(self.datasets['train'])==len(self.ogdatasets['train']):
                        self.best_valid_loss = epoch_loss
                        self.best_model_wts = copy.deepcopy(self.model.state_dict())

            if self.tb_logs is not None:
                print('getting lr')
                current_lr = self.optimizer.param_groups[0]['lr']
                self.tb.add_scalar('Learning Rate', current_lr, epoch)
                print('getting figs')
                if epoch%9==0:
                    train_fig = show_train_predictions(self.model,self.datasets['train'],self.device,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
                    print('got train fig')
                    valid_fig = show_valid_predictions(self.model,self.datasets['valid'],self.device,[60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75])
                    print('got valid fig')
                    self.tb.add_figure('train figs',train_fig)
                    self.tb.add_figure('valid figs',valid_fig)

            # if self.tb_logs is not None:
            #     for name, weight in self.model.named_parameters():
            #         if weight.grad is not None:
            #             self.tb.add_histogram(name, weight, epoch)
            print()

        print('Best valid loss: {:4f}'.format(self.best_valid_loss))
        # load best model weights
        # self.model.load_state_dict(best_model_wts)


    def sort_by_scores(self):
        self.temp_records = {}
        imgsize = self.ogdatasets['train'].img_size
        for i in range(len(self.ogdatasets['train'])):
            _filename = self.ogdatasets['train'].file_names[i]
            inputs,targets = self.ogdatasets['train'].data[_filename]
            inputs = inputs.float().view(-1, 3, imgsize, imgsize).to(self.device)
            targets_oneh = targets.view(-1, self.model.num_classes, imgsize, imgsize).to(self.device)
            targets = torch.argmax(targets_oneh, dim=1)
            with torch.no_grad():
                self.scoring_model.to(self.device)
                outputs = self.scoring_model(inputs)
                preds = torch.argmax(outputs, dim=1)
                loss = self.criterion(outputs, targets_oneh, targets)
                acc = torch.mean((targets == preds).to(float)).item()
                self.temp_records[_filename]=(loss.item(),acc)
        # self.scoring_model.to(torch.device('cpu'))
        self.ogdatasets['train'].data = {k:v for k,v in sorted(self.ogdatasets['train'].data.items(),key=lambda x:self.temp_records[x[0]][0],reverse=False)}
        self.ogdatasets['train'].file_names = sorted(self.ogdatasets['train'].file_names,key=lambda x:self.temp_records[x][0],reverse=False)


    def get_easy_dataset(self,size):
        self.datasets['train'].data = {}
        self.datasets['train'].file_names = []
        for i,k in enumerate(self.ogdatasets['train'].data.keys()):
            if i>=size:
                break
            self.datasets['train'].data[k] = self.ogdatasets['train'].data[k]
            self.datasets['train'].file_names.append(k)
        # temp_scores = []
        # for k in self.datasets['train'].data:
        #     temp_scores.append(self.temp_records[k][0])
        for key in self.ogdatasets:
            self.dataloaders[key] = DataLoader(self.datasets[key], batch_size=8, num_workers=6)
            self.dataset_sizes[key] = len(self.datasets[key])

    def train_manager(self,start_frac,epochs_per_step):
        self.best_model_weights = copy.deepcopy(self.model.state_dict())
        self.best_valid_loss=100.
        num_steps = len(epochs_per_step)
        tot_size = len(self.ogdatasets['train'])
        inc_fac = (1/start_frac)**(1.0/(num_steps-1))
        _size = tot_size*start_frac
        self.sort_by_scores()
        if self.tb_logs is not None:
            logpath = self.tb_logs['path']
            logcomment = self.tb_logs['comment']
            self.tb = SummaryWriter(log_dir=logpath, comment=logcomment)
        init_epoch = 0
        for step in range(num_steps):
            data_size = int(_size)
            self.get_easy_dataset(data_size)
            print('-'*100)
            print(f'Step # {step+1}')
            print('-'*100)
            for param_group in self.optimizer.param_groups:
                param_group['lr']=0.01
            if step==0:
                self.scheduler.patience = 30
            if step==1:
                self.scheduler.patience=15
            self.fit(list(range(init_epoch,init_epoch+epochs_per_step[step])))
            init_epoch += epochs_per_step[step]
            _size = _size*inc_fac
        self.tb.close()

def main():
    trainset = Prostate_data(img_size=256, num_classes=5)
    validset = Prostate_data(dataset_type='valid', img_size=256, num_classes=5)
    datasets = {'train': trainset, 'valid': validset}
    lr = 1e-2
    gamma = 0
    dprob = 0.2
    dtime = 'single_step_1731_1706'
    model = ResUnet(num_classes=5,dprob=dprob)
    optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    scoring_model = torch.load('logdirs/focal_fullruns_aug/respre_SGD_plateau/lr=0.01_gamma=0_dprob=0.2_0043_1606/0043_1606')
    params = {
        'datasets': datasets,
        'model': model,
        'criterion': Focalloss(gamma=gamma),
        'optimizer': optimizer,
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,min_lr=3e-6),
        'scoring_model': scoring_model,
        'tb_logs': {'path':f'logdirs/curriculum_rev_trials_aug/respre_SGD_plateau/lr={lr}_gamma={gamma}_dprob={dprob}_{dtime}','comment':f'lr={lr}_gamma={gamma}_dprob={dprob}_{dtime}'}
    }
    clearner = Curriculum_Learner(**params)
    try :
        clearner.train_manager(start_frac=0.3,epochs_per_step=[55,40])
        torch.save(clearner.model,f'logdirs/curriculum_rev_trials_aug/respre_SGD_plateau/lr={lr}_gamma={gamma}_dprob={dprob}_{dtime}/{dtime}')
        # show_valid_predictions(clearner.model, validset, clearner.device, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        # show_train_predictions(clearner.model, trainset, clearner.device, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    except Exception as e:
        print(e)
        torch.save(clearner.model,f'logdirs/curriculum_rev_trials_aug/respre_SGD_plateau/lr={lr}_gamma={gamma}_dprob={dprob}_{dtime}/{dtime}')
        # show_valid_predictions(clearner.model, validset, clearner.device,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        # show_train_predictions(clearner.model, trainset, clearner.device,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

if __name__ == '__main__':
    main()