from model_method import processDataset, llm_embedding, llm_embedding_2, emb_dic, splitTrainValTest, model_running, performance, changeOrder, prepareSourceTargetDataset,targetTune
from cnn_feedforward import CNN_feedforward
import torch

disease="FLU"
batch_size=32
epoch=50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

disease_label="delete M"  #"delete M"      #"M to F"
cui_label= "M to A"    #"M to A"
source='AC'
target='SLC'
freeze=True


df= processDataset(f"data/cleaned_{target}.xlsx", disease, disease_label, cui_label)
# df=changeOrder(df)

baselines=[True,False]
llms=['bert','biobert','medbert','openai_small','openai_large']
for baseline in baselines:
    for llm in llms:
        if baseline:
            llm='onehot'


        if baseline:
                cui2idx, embedding_matrix=emb_dic(df, baseline, None)
            # embedding_cui= llm_embedding_2("/Users/yuhe/Desktop/AIPH/weekly/2023/2023.12/embeddings_project8/UMLS_code_openai.h5", "/Users/yuhe/Desktop/AIPH/weekly/2023/2023.12/embeddings_project8/UMLS_code2_openai.h5")
        else:
            llm_emb=f"cui_embeddings/concept_pair_{llm}.h5"
            embedding_cui=llm_embedding(llm_emb)
            cui2idx, embedding_matrix=emb_dic(df,baseline, embedding_cui)



        # embedding_cui= llm_embedding_2("/Users/yuhe/Desktop/AIPH/weekly/2023/2023.12/embeddings_project8/UMLS_code_openai.h5", "/Users/yuhe/Desktop/AIPH/weekly/2023/2023.12/embeddings_project8/UMLS_code2_openai.h5")

        # print(embedding_matrix1.shape)
        # print(embedding_matrix2.shape)
        # print(cui2idx1)
        # print(cui2idx2)

        if baseline:
            inchannels=embedding_matrix.shape[0]
        else:
            inchannels=len(embedding_cui.columns)

        # source_train_loader, source_val_loader, target_train_loader, target_val_loader, target_test_loader, cui_channel=prepareSourceTargetDataset(df1,df2, disease, batch_size, cui2idx1, cui2idx2)
        # print(cui_channel)
        train_loader, val_loader, test_loader, cui_channel=splitTrainValTest(df, disease, batch_size, cui2idx)

        # seeds=[215]
        # seeds=[401]
        seeds=[215,401,114514]
        seeds_n=''
        for i in seeds:
            seeds_n+="_"+str(i)

        stride=1
        padding=0

        result_name=f"best_results/0913/freezeCL_{source}2{target}_{llm}_bs{batch_size}_e{epoch}_seed{seeds_n}"

        print(f"LLM {llm}")
        print("load model\n")
        # classifier, log=model_running(device, source_train_loader, source_val_loader,embedding_matrix1, cui2idx1, seeds, epoch, in_channels=inchannels, stride=stride, padding=padding,filter_sizes=[2,3,4],llm=llm, source=source)
        # classifier=model_running(device, train_loader, val_loader,embedding_matrix, cui2idx, seed, epoch, in_channels=len(embedding_cui.columns), stride=stride, padding=padding, filter_sizes=[cui_channel])  ## only 1 filter with width of all channel
        # classifier=model_running(device, train_loader, embedding_matrix, cui2idx, seed, epoch, in_channels=cui_channel, stride=stride, padding=padding)


        classifier = CNN_feedforward(pretrained_embedding=embedding_matrix, cuis_size=len(cui2idx), in_channels=inchannels, stride=stride, padding=padding, filter_sizes=[1],freezeCL=True)
        classifier.load_state_dict(torch.load(f'model_path/0913/{source}_best_model_{llm}.pth'))
        classifier.to(device)

        print("Starting tuning on target: ")
        tune_classifier,log_tune=targetTune(device, classifier, train_loader, val_loader, embedding_matrix, cui2idx, seeds, epoch,in_channels=inchannels, llm=llm, source=source, target=target,freezeCL='freezeCL_')

        print("############## Model Setting ###############")
        if baseline:
            print("llm: One-Hot")
        else:   
            print('llm: ', llm)
        print("Batch Size: ", batch_size)
        print("Epoch: ", epoch)
        print("Seed: ", seeds)
        print("Stride: ", stride)
        print("Padding: ", padding)
        print("Disease_label:", disease_label)
        print("CUI_label: ",cui_label)

        print("############## Model performance #############")
        print("test performance:")
        r1=performance(device, tune_classifier, test_loader, result_name+"_test.csv")
        print("train performance:")
        r2=performance(device, tune_classifier, train_loader,result_name+"_train.csv")

        log_name=f"experimentLog/0913/freezeCL_{source}2{target}_experiment_log_{llm}_bs{batch_size}_e{epoch}_seed{seeds_n}.txt"

        with open(log_name,'w') as f:
            f.write("############## TL Model Setting ###############\n")
            if baseline:
                f.write("llm: One-Hot\n")
            else:   
                f.write('llm: %s \n' % llm)
            f.write("Batch Size: %s \n" % batch_size)
            f.write("Epoch: %s \n" % epoch)
            f.write("Seed: %s \n" % seeds)
            f.write("Stride: %s \n" % stride)
            f.write("Padding: %s \n" % padding)
            f.write("Disease_label: %s \n" % disease_label)
            f.write("CUI_label %s: \n" % cui_label)

            f.write("\nLoad Model:\n")
            f.write(f'model_path/0913/{source}_best_model_{llm}.pth')

            f.write("\nStart Tuning Model:\n")
            f.write(log_tune)
            f.write("Tuning Complete.\n")

            f.write("\n############## Model performance #############\n")
            f.write("test performance: \n")
            f.write(r1)
            f.write("train performance: \n")
            f.write(r2)
        f.close()

        print(f"{llm} End")
        if baseline:
            break