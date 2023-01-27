# Prediction and Analysis of Intraoperative Hypotension Based on Ensemble Deep-learning

# Training model and validation model

# Author: Xian Tan (tanx431@nenu.edu.cn)



import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle, os, warnings
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve,recall_score,precision_score,matthews_corrcoef




warnings.filterwarnings("ignore")

# Database code part

class dataset(torch.utils.data.Dataset):
    def __init__(self, abp, ecg, ple, target):
        self.abp, self.ecg, self.ple= abp, ecg, ple
        self.target = target
    def __getitem__(self, index):
            a=np.array(self.abp[index])
            b=np.array(self.ecg[index])
            c=np.array(self.ple[index])
            return np.float32( np.vstack ((a,b,c)) ), np.float32(self.target[index])


    def __len__(self):
        return len(self.target)


# VGG NET

def Conv3x3BNReLU(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU6(inplace=True)
    )
class VGG(nn.Module):
    def __init__(self, block_nums):
        super(VGG, self).__init__()

        self.stage1 = self._make_layers(in_channels=3, out_channels=64, block_num=block_nums[0])
        self.stage2 = self._make_layers(in_channels=64, out_channels=128, block_num=block_nums[1])
        self.stage3 = self._make_layers(in_channels=128, out_channels=256, block_num=block_nums[2])
        self.stage4 = self._make_layers(in_channels=256, out_channels=512, block_num=block_nums[3])
        self.stage5 = self._make_layers(in_channels=512, out_channels=512, block_num=block_nums[4])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7,out_features=4096),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=1)
        )
        self.activation = nn.Sigmoid()

        self._init_params()

    def _make_layers(self, in_channels, out_channels, block_num):
        layers = []
        layers.append(Conv3x3BNReLU(in_channels,out_channels))
        for i in range(1,block_num):
            layers.append(Conv3x3BNReLU(out_channels,out_channels))
        layers.append(nn.MaxPool1d(kernel_size=2,stride=2, ceil_mode=False))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = x.view(x.size(0),-1)
        out = self.classifier(x)
        out = self.activation(out)
        return out

# Deep model of Lee et al.

class Net(nn.Module):
    def __init__(self, dr):

        super(Net, self).__init__()
        self.dr = dr
        self.final=1

        self.inc = 3

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.inc, out_channels=64, kernel_size=10, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(self.dr)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(self.dr)
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(self.dr)
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(self.dr)
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(self.dr)
        )

        self.fc = nn.Sequential(
            nn.Linear(320, self.final),
            nn.Dropout(self.dr)
        )

        self.activation = nn.Sigmoid()

    
    def forward(self, x):

        x = x.view(x.shape[0], self.inc, -1)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = out.view(x.shape[0], out.size(1) * out.size(2))

        out = self.fc(out)
        out = self.activation(out)

        return out

class EDIHP(nn.Module):
    def __init__(self, dr, N_classifiers=3):
        """
        :param dr: Dropout rate
        :param N_classifiers: Classifiers number of ensemble model
        """
        super(EDIHP, self).__init__()
        self.N_classifiers = N_classifiers
        self.dr = dr
        self.final = 1
        self.inc = 3
        self.conv = []
        for n in range(N_classifiers):
            setattr(self,"conv"+str(n),nn.Sequential(
                nn.Conv1d(in_channels=self.inc, out_channels=64, kernel_size=10, stride=1, padding=0),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),
                nn.Dropout(self.dr),
                nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),
                nn.Dropout(self.dr),
                nn.Conv1d(128, 64, kernel_size=16, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),
                nn.Dropout(self.dr),
                nn.Conv1d(64, 32, kernel_size=16, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),
                nn.Dropout(self.dr),
                nn.Conv1d(32, 16, kernel_size=16, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),
                nn.Dropout(self.dr)

                ))
            setattr(self,"LSTM"+str(n), nn.Sequential(
                nn.LSTM(input_size=1296,hidden_size=128)
            ))
        self.fc = nn.Sequential(
            nn.Linear(128, self.final)
        )

        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = x.view(x.shape[0], self.inc, -1)
        out = []
        for i in range(self.N_classifiers):
            out_t=getattr(self, "conv" + str(i))(x)
            out_t=out_t.view(x.shape[0],1296)
            out_t,_=getattr(self, "LSTM" + str(i))(out_t)
            out.append(out_t)
        ensemble = 0
        for o in out:
            if isinstance(ensemble, int):
                ensemble = o
            else:
                ensemble =ensemble + o
        ensemble = ensemble.view(x.shape[0], 128)
        out=self.fc(ensemble)
        out = self.activation(out)

        return out

#CNN model


class CNN(nn.Module):
    def __init__(self, dr):
        super(CNN, self).__init__()
        self.dr = dr
        self.final = 1
        self.inc = 3
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.inc, out_channels=64, kernel_size=10, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr),
            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr),
            nn.Conv1d(128, 1, kernel_size=16, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr),
            nn.Linear(364,1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.inc, out_channels=64, kernel_size=10, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr),
            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr),
            nn.Conv1d(128, 1, kernel_size=16, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr),
            nn.Linear(364, 1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.inc, out_channels=64, kernel_size=10, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr),
            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr),
            nn.Conv1d(128, 1, kernel_size=16, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr),
            nn.Linear(364, 1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=self.inc, out_channels=64, kernel_size=10, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr),
            nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr),
            nn.Conv1d(128, 1, kernel_size=16, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(self.dr),
            nn.Linear(364, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(1, self.final),
            nn.Dropout(self.dr)
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):

        x = x.view(x.shape[0], self.inc, -1)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out=out1+out2+out3+out4
        out = out.view(x.shape[0], 1)

        out = self.fc(out)
        if self.task == 'classification':
            out = self.activation(out)

        return out

# AlexNet

class AlexNet(nn.Module):
    def __init__(self,num_classes=1):
        super(AlexNet,self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv1d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=2,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
            nn.Conv1d(in_channels=96,out_channels=192,kernel_size=5,stride=1,padding=2,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
            nn.Conv1d(in_channels=192,out_channels=384,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256*6*6,out_features=4096),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
    def forward(self,x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0),256*6*6)
        x = self.classifier(x)
        return x
# Load Dataset

def binary(list):
    list2=[]
    for i in list:
        if i>=0.5:
            list2.append(1)
        else:
            list2.append(0)
    return list2

def main(N_classifiers=None):
    """
    :param N_classifiers: Classifiers number of ensemble model, 'none' to test the model of Lee et al.
    :return: None
    """
    forecast= 900
    batch_size = 128
    max_epoch = 200
    list=['abp','ecg','ple']
    random_key = 12127
    dr = 0.3
    load_data = './Data/'
    datalist = np.char.split(np.array(os.listdir(load_data)), '.')
    case_data = []
    for caseid in datalist:
        case_data.append ( int ( caseid[0] ) )
    print ( 'The number of case samples is {}'.format ( len ( case_data ) ) )

    assort = {}
    assort['train'], assort['valid+test'] = train_test_split ( case_data,
                                                        test_size=(0.4),
                                                        random_state=random_key )
    assort['valid'], assort['test'] = train_test_split ( assort['valid+test'],
                                                  test_size=(0.1/0.4),
                                                  random_state=random_key )

    for stage in [ 'train', 'valid', 'test' ]:
        print ( "{}:{}".format(stage, len(assort[stage])) )
    for idx, caseid in enumerate(case_data):
        filename = load_data + str ( caseid ) + '.pkl'#load pkl
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
            for j in list:
                for i in range(len(data[j])):
                    if len(data[j][i]) != 3000:
                        print("caseid", caseid, len(data[j][i]))
                    for k in range(len(data[j][i])):

                        if math.isnan(data[j][i][k]):
                            data[j][i][k]=0
            data['caseid'] = [ caseid ] * len ( data['abp'] )
  
            raw_records = raw_records.append ( pd.DataFrame ( data ) ) if idx > 0 else pd.DataFrame ( data )
        
    raw_records = raw_records[(raw_records['map']>=20)&(raw_records['map']<=160)].reset_index(drop=True)


    # Build model

    task_target = 'hypo'
    criterion = nn.BCELoss()
    records = {}
    elt = {}
    for stage in [ 'train', 'valid', 'test' ]:
        elt[stage] = {}
        for x in [ 'abp', 'ecg', 'ple','hypo', 'map' ]:
            elt[stage][x] = records[stage][x]

    dataset, loader = {}, {}
    epoch_loss, epoch_auc = {}, {}
    for stage in [ 'train', 'valid', 'test' ]:
        dataset[stage] = dataset(elt[stage]['abp'], elt[stage]['ecg'], elt[stage]['ple'], elt[stage][task_target])
        loader[stage] = torch.utils.data.DataLoader(dataset[stage],
                                                batch_size=batch_size,
                                                shuffle = True if stage == 'train' else False )
        epoch_loss[stage], epoch_auc[stage] = [], []

    # Training and validation

    if N_classifiers!=None:
        STF = EDIHP(dr=dr,N_classifiers=N_classifiers)
    else:
        STF = Net(dr=dr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    STF = STF.to(device)
    optimizer = torch.optim.Adam(STF.parameters(), lr=0.0005)
    n_epochs = max_epoch
    top_loss, top_auc = 99999.99999, 0.0
    for epoch in range(n_epochs): 

        target_stack, output_stack = {}, {}
        output_stack_binary = {}
        latest_loss, latest_auc = {}, {}
        latest_acc, latest_pre, latest_spe, latest_sen, latest_mcc = {}, {}, {}, {}, {}
        for stage in [ 'train', 'valid', 'test' ]:
            target_stack[stage], output_stack[stage] =  [], []
            latest_loss[stage], latest_auc[stage] = 0.0, 0.0

        STF.train()

        for stf_inputs, stf_target in loader['train']:

            stf_inputs, stf_target = stf_inputs.to(device), stf_target.to(device)
            optimizer.zero_grad()
            stf_output = STF( stf_inputs )

            loss = criterion(stf_output.T[0], stf_target)
            latest_loss['train'] += loss.item()*stf_inputs.size(0)

            loss.backward()
            optimizer.step()

        latest_loss['train'] = latest_loss['train']/len(loader['train'].dataset)
        epoch_loss['train'].append ( latest_loss['train'] )

        for stage in [ 'valid', 'test']:
    
            STF.eval()
            with torch.no_grad():
                for stf_inputs, stf_target in loader[stage]:

                    stf_inputs, stf_target = stf_inputs.to(device), stf_target.to(device)
                    stf_output = STF( stf_inputs )
                    target_stack[stage].extend ( np.array ( stf_target.cpu() ) )
                    output_stack[stage].extend ( np.array ( stf_output.cpu().T[0] ) )

                    loss = criterion(stf_output.T[0], stf_target)
                    latest_loss[stage] += loss.item()*stf_inputs.size(0)

                latest_loss[stage] = latest_loss[stage]/len(loader[stage].dataset)
                epoch_loss[stage].append ( latest_loss[stage] )

        for stage in ['valid', 'test']:
            latest_auc[stage] = roc_auc_score ( target_stack[stage], output_stack[stage] )
            output_stack_binary[stage] = binary(output_stack[stage])
            latest_acc[stage] = accuracy_score(target_stack[stage], output_stack_binary[stage])
            latest_pre[stage] = precision_score(target_stack[stage], output_stack_binary[stage])
            latest_sen[stage] = recall_score(target_stack[stage], output_stack_binary[stage])
            latest_mcc[stage] = matthews_corrcoef(target_stack[stage], output_stack_binary[stage])
            tn, fp, fn, tp = confusion_matrix(target_stack[stage], output_stack_binary[stage]).ravel()
            latest_spe[stage] = tn / (tn + fp)
            epoch_auc[stage].append ( latest_auc[stage] )

        if abs(latest_auc['valid']) > abs(top_auc):
            top_auc = abs(latest_auc['valid'])
        print ( 'Train loss: {:.4f} / Valid loss: {:.4f} (AUC: {:.4f}) / Test loss: {:.4f} (AUC: {:.4f})'.format
            ( latest_loss['train'],latest_loss['valid'], latest_auc['valid'],latest_loss['test'], latest_auc['test']) )
        print("acc", latest_acc)
        print("pre", latest_pre)
        print("sen", latest_sen)
        print("spe", latest_spe)
        print("mcc", latest_mcc)
if __name__=="__main__":
    main(7)
