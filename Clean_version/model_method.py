import numpy as np
import pandas as pd
import torch
import h5py
from class_define import Dataset, AverageMeter, ForeverDataIterator
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import SGD, Adam
from cnn_feedforward import CNN_feedforward
import time
from sklearn import metrics
from sklearn.metrics import roc_auc_score
def processDataset(file_path,disease, disease_label, cui_label=None, source=False, upsampling=False):
    #read data
    df=pd.read_excel(file_path)

    if disease_label=="M to F":
    #update M to F
        df[disease+'_diagnosis']=np.where(df[disease+'_diagnosis']=='M', 'F', df[disease+'_diagnosis'])
    elif disease_label =="delete M":
    #delete M diagnosis
        df=df[df[disease+'_diagnosis']!='M']

    if cui_label=="M to A":
    #update M to F
        for col in df.loc[:,df.columns.str.startswith("C")].columns:
            if col =="C0424781":
                continue
            df[col]=np.where(df[col]=='M', 'A', df[col])
        for col in df.loc[:,df.columns.str.startswith("AC")].columns:
            df[col]=np.where(df[col]=='M', 'A', df[col])
        for col in df.loc[:,df.columns.str.startswith("OC")].columns:
            df[col]=np.where(df[col]=='M', 'A', df[col])
    
    if source:
        df=df[df['admityear']!='20140601-20150531']

    if upsampling:
        neg=df[df[disease+'_diagnosis']=='F']
        pos=df[df[disease+'_diagnosis']=='T']
        ratio=neg.shape[0]/pos.shape[0]
        for i in range(int(ratio)):
            neg=pd.concat([neg,pos],ignore_index=True)
        df=neg
    
    return df

def changeOrder(df):
    temp=df.loc[:, df.columns.str.startswith("C") | df.columns.str.startswith("AC") | df.columns.str.startswith("OC")]
    rank={i:0  for i in temp.columns}
    for cui in temp.columns:
        # print(temp[temp[cui]=="P"])
        rank[cui]=temp[temp[cui]=="P"].shape[0]
    rank=dict(sorted(rank.items(), key=lambda x:x[1], reverse=True))

    l1=[i for i in df.columns if i not in temp.columns]
    l1[1:1]=rank.keys()
    df=df[l1]
    return df

def get_APM_cui(df, df2=pd.DataFrame()):
    df=df.loc[:, df.columns.str.startswith("C") | df.columns.str.startswith("AC") | df.columns.str.startswith("OC") | df.columns.str.startswith("age")]
    if df2.empty:
        return df, pd.get_dummies(df).columns
    else:
        df2=df2.loc[:, df2.columns.str.startswith("C") | df2.columns.str.startswith("AC") | df2.columns.str.startswith("OC") | df2.columns.str.startswith("age")]
        return df, set(pd.get_dummies(df2).columns.to_list()+pd.get_dummies(df).columns.to_list())
    
def emb_dic(df,baseline=False,embedding_df=None,df2=pd.DataFrame()):
    l=get_APM_cui(df,df2)[1]
    l=sorted(l)

    cuis=[]
    idx=0
    cui2idx={}

    for i in l:
        cui=i
        cuis.append(cui)
        # cui2idx[cui]=idx      # wo. mask
        # idx+=1

        if 'M' not in cui:      # w. mask
            idx+=1
            cui2idx[cui]=idx
            
        else:
            cui2idx[cui]=0

    # matrix_len=len(l) # wo. mask
    matrix_len=idx+1

    if baseline:
        print("One-hot embedding")
        embedding_matrix = np.zeros((matrix_len, matrix_len))
        for i in cui2idx:
            if 'M' in i:
                continue
            embedding_matrix[cui2idx[i]][cui2idx[i]]=1

    else:
        print('Semnatic embedding')                                
        embedding_matrix = np.zeros((matrix_len, len(embedding_df.columns)))
        words_found = 0

        for row in embedding_df.itertuples():
            try: 
                embedding_matrix[cui2idx[row[0]]] = row[1:]
                words_found += 1
                # print(row[0])
            except KeyError:
                continue
        print(words_found)
    return cui2idx, torch.Tensor(embedding_matrix)

def df2ids(df, cui2idx):
    df=get_APM_cui(df)[0]
    col=df.columns # wo. mask
    # col=[i for i in df.columns if 'M' not in i] # w. mask
    input_ids=[]

    for row in df.itertuples():
        input_id=[]
        for i,j in enumerate(col):
            input_id.append(cui2idx[j+'_'+row[1+i]])
        input_ids.append(input_id)
        
    return pd.DataFrame(input_ids)


def llm_embedding(h5path):
    cui_dict={}
    with h5py.File(h5path, "r") as f:
        n=0
        v_len=len(f[list(f.keys())[0]][()][0])
    #     print(v_len)
        # print(len(list(f.keys())))
        for key in list(f.keys()):
            if "agegroup" in key.split()[0]:
                cui_dict[key.split()[0]]=list(f[key])[0]
            elif "C0424781" in key.split()[0]:
                cui_dict[key.split()[0]]=list(f[key])[0]
            elif key.split()[1]=="Not":
                cui_dict[key.split()[0]+"_A"]=list(f[key])[0]
            else:
                cui_dict[key.split()[0]+"_P"]=list(f[key])[0]
            n+=1
            # if n%10000==0:
            #     print(n)
    return pd.DataFrame.from_dict(cui_dict, orient='index', columns=['V'+ str(i) for i in range(v_len)])

def llm_embedding_2(h5path1,h5path2):
    cui_dict={}

    with h5py.File(h5path1, "r") as f:
        n=0
        v_len=len(f[list(f.keys())[0]][()][0])
    #     print(v_len)
        # print(len(list(f.keys())))
        for key in list(f.keys()):
            if key.split()[1]=="Not":
                cui_dict[key.split()[0]+"_A"]=list(f[key])[0]
            else:
                cui_dict[key.split()[0]+"_P"]=list(f[key])[0]
            n+=1
            # if n%10000==0:
            #     print(n)
    with h5py.File(h5path2, "r") as f:
        n=0
        v_len=len(f[list(f.keys())[0]][()][0])
    #     print(v_len)
        # print(len(list(f.keys())))
        for key in list(f.keys()):
            if key.split()[1]=="Not":
                cui_dict[key.split()[0]+"_A"]=list(f[key])[0]
            else:
                cui_dict[key.split()[0]+"_P"]=list(f[key])[0]
            n+=1
            # if n%10000==0:
            #     print(n)

    embedding_cui=pd.DataFrame.from_dict(cui_dict, orient='index', columns=['V'+ str(i) for i in range(v_len)])
    return embedding_cui

def splitTrainValTest(df, disease, batch_size,cui2idx):
    df[disease+'_diagnosis'] = pd.Categorical(df[disease+'_diagnosis']) # set label to Categorical for separate processing
    df[disease+'_diagnosis'+ "-codes"] = df[disease+'_diagnosis'].cat.codes # add codes column for label (e.g. 0, 1, 2)

    #split train and test
    train_val=df[~df['admityear'].isin(['20140601-20150531'])]
    # train=train_val[~train_val['admityear'].isin(['20130601-20140531'])]
    # val=train_val[train_val['admityear'].isin(['20130601-20140531'])]
    test=df[df['admityear'].isin(['20140601-20150531'])]
    train=train_val.sample(frac=0.8, random_state=1)
    val = train_val.drop(train.index)
    # val=train_val[~train_val["ID"].isin(train["ID"])]
    print(train[disease+'_diagnosis'].value_counts())
    print(val[disease+'_diagnosis'].value_counts())

    # print(test)

    train_dataset=Dataset(df2ids(train, cui2idx), train[disease+'_diagnosis-codes'])
    val_dataset=Dataset(df2ids(val, cui2idx), val[disease+'_diagnosis-codes'])
    test_dataset=Dataset(df2ids(test, cui2idx), test[disease+'_diagnosis-codes'])

    train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    cui_channel=df2ids(train, cui2idx).shape[1]
    return train_loader, val_loader, test_loader, cui_channel

def prepareSourceTargetDataset(df1,df2, disease, batch_size,cui2idx1, cui2idx2):
    df1[disease+'_diagnosis'] = pd.Categorical(df1[disease+'_diagnosis']) # set label to Categorical for separate processing
    df1[disease+'_diagnosis'+ "-codes"] = df1[disease+'_diagnosis'].cat.codes # add codes column for label (e.g. 0, 1, 2)

    df2[disease+'_diagnosis'] = pd.Categorical(df2[disease+'_diagnosis']) # set label to Categorical for separate processing
    df2[disease+'_diagnosis'+ "-codes"] = df2[disease+'_diagnosis'].cat.codes # add codes column for label (e.g. 0, 1, 2)
    #split train and test

    # source_train=df1[~df1['admityear'].isin(['20140601-20150531'])]
    # source_val=df1[df1['admityear'].isin(['20140601-20150531'])]
    source_train=df1.sample(frac=0.8, random_state=1)
    source_val=df1[~df1["ID"].isin(source_train["ID"])]

    target_train_val=df2[~df2['admityear'].isin(['20140601-20150531'])]
    # target_train=target_train_val[~target_train_val['admityear'].isin(['20130601-20140531'])]
    # target_val=target_train_val[target_train_val['admityear'].isin(['20130601-20140531'])]
    target_test=df2[df2['admityear'].isin(['20140601-20150531'])]
    target_train=target_train_val.sample(frac=0.8, random_state=1)
    target_val=target_train_val[~target_train_val["ID"].isin(target_train["ID"])]
    

    # print(test)

    source_train_dataset=Dataset(df2ids(source_train, cui2idx1), source_train[disease+'_diagnosis-codes'])
    source_val_dataset=Dataset(df2ids(source_val, cui2idx1), source_val[disease+'_diagnosis-codes'])
    target_train_dataset=Dataset(df2ids(target_train, cui2idx2), target_train[disease+'_diagnosis-codes'])
    target_val_dataset=Dataset(df2ids(target_val, cui2idx2), target_val[disease+'_diagnosis-codes'])
    target_test_dataset=Dataset(df2ids(target_test, cui2idx2), target_test[disease+'_diagnosis-codes'])

    source_train_loader=DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    source_val_loader=DataLoader(source_val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_train_loader=DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_val_loader=DataLoader(target_val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_test_loader=DataLoader(target_test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    cui_channel=df2ids(source_train, cui2idx1).shape[1]
    return source_train_loader, source_val_loader, target_train_loader,target_val_loader,target_test_loader, cui_channel


def model_running(device, train_loader,val_loader, embedding_matrix, cui2idx, seeds, epoch, in_channels, stride, padding, filter_sizes,llm, source):
    log=""
    best_auroc = 0.
    for seed in seeds:
        torch.manual_seed(seed)

        classifier = CNN_feedforward(pretrained_embedding=embedding_matrix, cuis_size=len(cui2idx), in_channels=in_channels, stride=stride, padding=padding, filter_sizes=filter_sizes)
        classifier.to(device)

        # define optimizer and lr scheduler
        optimizer = SGD(classifier.parameters(), 0.001, momentum=0.9, weight_decay=1e-3, nesterov=True)

        # optimizer = Adam(classifier.parameters(), lr=0.001, weight_decay=1e-3)

        train_loader_iter=ForeverDataIterator(train_loader)
        # start training
        

        for e in range(epoch): #10 epoch
            classifier.train()
            batch_time = AverageMeter('Time', ':5.2f')
            data_time = AverageMeter('Data', ':5.2f')
            losses = AverageMeter('Loss', ':6.2f')
            end = time.time()
            for i in range(100): # 100 iter per epoch
                # optimizer.step()
            
                # measure data loading time
                data_time.update(time.time() - end)
            
                x_s, labels_s = next(train_loader_iter)
                # print("x_s: ",x_s.shape)
                x_s = x_s.to(device)
                labels_s = labels_s.to(device)
            
                # compute output
                y_s = classifier(x_s,True)
                loss = F.cross_entropy(y_s, labels_s)
            
                # update meters
                losses.update(loss.item(), x_s.size(0))
            
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
        
            classifier.eval()
            val_losses = AverageMeter('Val Loss', ':6.2f')
            val_correct = 0
            val_total = 0
            
            all_labels=[]
            all_pred=[]

            with torch.no_grad():
                for x_val, labels_val in val_loader:
                    x_val = x_val.to(device)
                    labels_val = labels_val.to(device)
                    
                    y_val = classifier(x_val)
                    val_loss = F.cross_entropy(y_val, labels_val)
                    
                    val_losses.update(val_loss.item(), x_val.size(0))
                    
                    _, predicted = torch.max(y_val, 1)
                    val_correct += (predicted == labels_val).sum().item()
                    val_total += labels_val.size(0)

                    all_labels.extend(labels_val.cpu().numpy())
                    all_pred.extend(y_val.cpu().numpy())
                    # print(y_val)
            
            val_accuracy = val_correct / val_total
            # print(all_labels)
            # print(all_pred[:][1])
            all_pred=np.array(all_pred)
            AUROC=roc_auc_score(all_labels, all_pred[:,1])
            print(f'Seed: {seed}, Epoch [{e+1}/{epoch}], Train Loss: {losses.avg:.4f}, Val Loss: {val_losses.avg:.4f}, Val Accuracy: {val_accuracy:.4f}, Val AUROC: {AUROC:.4f}')
            log+=f'Seed: {seed}, Epoch [{e+1}/{epoch}], Train Loss: {losses.avg:.4f}, Val Loss: {val_losses.avg:.4f}, Val Accuracy: {val_accuracy:.4f}, Val AUROC: {AUROC:.4f}\n'
            
            # Save the best model
            # if val_accuracy > best_acc1:
            #     best_acc1 = val_accuracy
            #     torch.save(classifier.state_dict(), 'best_model.pth')
            if AUROC > best_auroc:
                best_auroc = AUROC
                torch.save(classifier.state_dict(), f'model_path/{source}_best_model_{llm}.pth')

    print("Training complete.")
    print(classifier.load_state_dict(torch.load( f'model_path/{source}_best_model_{llm}.pth')))

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in classifier.state_dict():
        print(param_tensor, "\t", classifier.state_dict()[param_tensor].size())
    
    return classifier, log


def targetTune(device,classifier, target_train_loader, target_val_loader,embedding_matrix,cui2idx, seeds,epoch,in_channels,llm, source, target, freezeCL):
    log=''
    best_acc1 = 0.
    best_auroc=0.
    embedding_matrix=embedding_matrix.to(device)
    for seed in seeds:
        torch.manual_seed(seed)
        # classifier = CNN_feedforward(pretrained_embedding=embedding_matrix, cuis_size=len(cui2idx), in_channels=in_channels, stride=1, padding=0, filter_sizes=[2,3,4])
        # classifier.load_state_dict(torch.load(classifier_path))
        classifier.to(device)

        classifier.embedding=nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        # define optimizer and lr scheduler
        optimizer = SGD(classifier.parameters(), 0.01, momentum=0.9, weight_decay=1e-3, nesterov=True)

        train_loader_iter=ForeverDataIterator(target_train_loader)
        # start training
        

        for e in range(epoch): #10 epoch
            classifier.train()
            batch_time = AverageMeter('Time', ':5.2f')
            data_time = AverageMeter('Data', ':5.2f')
            losses = AverageMeter('Loss', ':6.2f')
            end = time.time()
            for i in range(100): # 100 iter per epoch
                # optimizer.step()
            
                # measure data loading time
                data_time.update(time.time() - end)
            
                x_s, labels_s = next(train_loader_iter)
                x_s = x_s.to(device)
                labels_s = labels_s.to(device)

                # compute output
                y_s = classifier(x_s, True)
                loss = F.cross_entropy(y_s, labels_s)
            
                # update meters
                losses.update(loss.item(), x_s.size(0))
            
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
        
            classifier.eval()
            val_losses = AverageMeter('Val Loss', ':6.2f')
            val_correct = 0
            val_total = 0
            all_labels=[]
            all_pred=[]
            
            with torch.no_grad():
                for x_val, labels_val in target_val_loader:
                    x_val = x_val.to(device)
                    labels_val = labels_val.to(device)
                    
                    y_val = classifier(x_val)
                    val_loss = F.cross_entropy(y_val, labels_val)
                    
                    val_losses.update(val_loss.item(), x_val.size(0))
                    
                    _, predicted = torch.max(y_val, 1)
                    val_correct += (predicted == labels_val).sum().item()
                    val_total += labels_val.size(0)

                    all_labels.extend(labels_val.cpu().numpy())
                    all_pred.extend(y_val.cpu().numpy())
            
            val_accuracy = val_correct / val_total

            all_pred=np.array(all_pred)
            AUROC=roc_auc_score(all_labels, all_pred[:,1])
            print(f'Seed: {seed}, Epoch [{e+1}/{epoch}], Train Loss: {losses.avg:.4f}, Val Loss: {val_losses.avg:.4f}, Val Accuracy: {val_accuracy:.4f}')
            log+=f'Seed: {seed}, Epoch [{e+1}/{epoch}], Train Loss: {losses.avg:.4f}, Val Loss: {val_losses.avg:.4f}, Val Accuracy: {val_accuracy:.4f}\n'
            
            # Save the best model
            # if val_accuracy > best_acc1:
            #     best_acc1 = val_accuracy
            #     torch.save(classifier.state_dict(), 'best_model_afterTargetTune.pth')
            if AUROC > best_auroc:
                best_auroc = AUROC
                torch.save(classifier.state_dict(), f'model_path/{freezeCL}{source}2{target}_best_model_{llm}.pth')

    print("Tuning complete.")
    print(classifier.load_state_dict(torch.load(f'model_path/{freezeCL}{source}2{target}_best_model_{llm}.pth')))

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in classifier.state_dict():
        print(param_tensor, "\t", classifier.state_dict()[param_tensor].size())
    
    return classifier, log
    

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def printListToFile(fileName, total_y_true, total_y_pred1, total_y_pred2, 
                    total_y_diagnosis, total_y_correct):
    a = np.asarray(total_y_true).astype(int)
    b1 = np.asarray(total_y_pred1)
    b2 = np.asarray(total_y_pred2)

    c = np.asarray(total_y_diagnosis).astype(int)
    d = np.asarray(total_y_correct).astype(int)
    df = pd.DataFrame({"y_true": a, "p0": b1, "p1": b2, "prediction": c, "correct": d})
    df.to_csv(fileName, index=False)

    # calculate auc
    # positive: 1, negative:0
    df['I_category'] = 'F'
    df.loc[df['y_true'] == 0, "I_category"] = "T"
    df['M_category'] = 'F'
    df.loc[df['y_true'] == 1, "M_category"] = "T"
   
    fpr, tpr, thresholds = metrics.roc_curve(df['I_category'], df['p0'], pos_label='T')
    auc_I = metrics.auc(fpr, tpr)
    fpr, tpr, thresholds = metrics.roc_curve(df['M_category'], df['p1'], pos_label='T')
    auc_M = metrics.auc(fpr, tpr)
    
    avg_auc = (auc_I + auc_M) / 2
    return avg_auc, auc_I, auc_M


def performance(device, classifier, loader, file_name):
    # torch.manual_seed(seed)
    total_y_pred1 = np.array([[]])
    total_y_pred2 = np.array([[]])
    
    total_y_diagnosis = np.array([[]])
    total_y_correct = np.array([[]])
    total_y_true = np.array([])
    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    classifier.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (features, target) in enumerate(loader):
            features = features.to(device)
            target = target.to(device)
    
            # compute output
            output = classifier(features)
            loss = F.cross_entropy(output, target)
    
            # measure accuracy and record loss
            acc1 = accuracy(output, target)
            losses.update(loss.item(), features.size(0))
            top1.update(acc1[0].item(), features.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            probAll = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
            total_y_pred1 = np.append(total_y_pred1, probAll[:, 0])
            total_y_pred2 = np.append(total_y_pred2, probAll[:, 1])
    
            total_y_true = np.append(total_y_true, target.cpu().numpy())
            _, diagnosis = output.topk(1, 1, True, True)
            diagnosis = diagnosis.t()
            total_y_diagnosis = np.append(total_y_diagnosis, diagnosis.cpu().numpy())
            correct = diagnosis.eq(target.view(1, -1).expand_as(diagnosis))
            total_y_correct = np.append(total_y_correct, correct.cpu().numpy())
    
        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))
    
        avg_auc,auc_I,auc_M= printListToFile(file_name,total_y_true,total_y_pred1,total_y_pred2, total_y_diagnosis,total_y_correct)
        print(f' AUROC {avg_auc}')
        return ' * Acc@1 {top1.avg:.3f}'.format(top1=top1)+"\n"+f' * AUROC {avg_auc}\n '

