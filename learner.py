from torch.utils.data import DataLoader
import time
import torch
from tqdm import tqdm
import copy
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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
        0: [0, 255, 0],
        1: [0, 0, 255],
        2: [255, 255, 255],
        3: [255, 0, 0],
        4: [255, 255, 0]
    }
    img_h = tensor_img.size()[2]
    out_img = np.zeros((img_h,img_h,3))
    for h in range(img_h):
        for w in range(img_h):
            pixel_class = torch.argmax(tensor_img[:,h,w]).item()
            out_img[h,w,:] = pallete_dict[pixel_class]
    final_img = Image.fromarray(out_img.astype('uint8'), 'RGB')
    return final_img

class Learner():

    def __init__(self,datasets,model,criterion,optimizer,scheduler=None,bs=4,num_workers=2,device='cuda:0'):
        self.datasets = datasets
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = {}
        self.dataset_sizes = {}
        for key in self.datasets:
            shuffle = True if key=='train' else False
            self.dataloaders[key] = DataLoader(self.datasets[key],batch_size=bs,num_workers=num_workers,shuffle=shuffle)
            self.dataset_sizes[key] = len(self.datasets[key])


    def find_lr(self,max_lr=10,stop_div=True):
        pass

    def fit(self,tb_logs=None,epochs=1):
        '''
        :param tb_logs: a dictionary containing tensorboard log flag, path and comment
        :param epochs:
        :return:
        '''

        imgsize = self.datasets['train'].img_size
        if tb_logs is not None:
            logpath = tb_logs['path']
            logcomment = tb_logs['comment']
            tb = SummaryWriter(log_dir=logpath+f'/{logcomment}', comment=logcomment)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_valid_loss=100.
        self.record_dict = {'train': {'loss': [], 'acc': []}, 'valid': {'loss': [], 'acc': []}}
        for epoch in range(epochs):
            print(f'EPOCH : {epoch + 1}/{epochs}')
            for phase in self.dataloaders.keys():
                since = time.time()
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                running_loss = 0.
                running_acc = 0.
                for inputs, targets in tqdm(self.dataloaders[phase]):
                    inputs = inputs.float().view(-1, 3, imgsize, imgsize).to(self.device)
                    targets_oneh = targets.view(-1, self.model.num_classes, imgsize, imgsize).to(self.device)
                    targets = torch.argmax(targets_oneh,dim=1)
                    with torch.set_grad_enabled(phase=='train'):
                        self.optimizer.zero_grad()
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs,dim=1)
                        loss = self.criterion(outputs, targets_oneh, targets)
                        acc = torch.mean((targets==preds).to(float)).item()
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                        running_loss += loss.item() * inputs.size()[0]
                        running_acc += acc*inputs.size()[0]
                #                 if phase == 'train' and not(scheduler is None):
                #                     scheduler.step(epoch_loss)

                epoch_loss = running_loss/self.dataset_sizes[phase]
                epoch_acc = running_acc/self.dataset_sizes[phase]

                if epoch_acc>0.65:
                    for param_groups in self.optimizer.param_groups:
                        param_groups['lr']=0.0001

                if tb_logs is not None:
                    if phase == 'train':
                        tb.add_scalar('Train Loss', epoch_loss, epoch)
                        tb.add_scalar('Train Acc',epoch_acc,epoch)
                    else:
                        tb.add_scalar('Valid Loss', epoch_loss, epoch)
                        tb.add_scalar('Valid Acc', epoch_acc, epoch)

                if phase == 'valid' and not (self.scheduler is None):  # FOR SINGLE SAMPLE
                    self.scheduler.step(epoch_loss)

                self.record_dict[phase]['loss'].append(epoch_loss)
                self.record_dict[phase]['acc'].append(epoch_acc)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,epoch_acc))
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                # deep copy the model
                if phase == 'valid' and epoch_loss < best_valid_loss:
                    best_valid_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            if tb_logs is not None:
                current_lr = self.optimizer.param_groups[0]['lr']
                tb.add_scalar('Learning Rate', current_lr, epoch)
                if epoch%15==0:
                    print('getting figures')
                    train_fig = show_train_predictions(self.model,self.datasets['train'],self.device,[20,21,22,23,24,25,26,27,28,29,30,31,32,33])
                    valid_fig = show_valid_predictions(self.model,self.datasets['valid'],self.device,[50,51,52,53,54,55,56,57,58,59,60,61,62,63])
                    tb.add_figure('train figs',train_fig)
                    tb.add_figure('valid figs',valid_fig)

            print()

        print('Best valid loss: {:4f}'.format(best_valid_loss))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        if tb_logs is not None:
            tb.close()
