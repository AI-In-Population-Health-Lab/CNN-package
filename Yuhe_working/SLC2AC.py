from model_method import processDataset, llm_embedding, llm_embedding_2, emb_dic, splitTrainValTest, model_running, performance, changeOrder, prepareSourceTargetDataset,targetTune
import torch

disease="FLU"
batch_size=32
epoch=50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

disease_label="delete M"  #"delete M"      #"M to F"
cui_label= None    #"M to A"


df2= processDataset("/Users/yuhe/Desktop/AIPH/weekly/2024/March/AC_small.xlsx", disease, disease_label, cui_label)
df1= processDataset("/Users/yuhe/Desktop/AIPH/weekly/2024/March/cleaned_SLC.xlsx", disease, disease_label, cui_label)
# df=changeOrder(df)


baseline=False
llms=['bert','biobert','medbert','openai_small','openai_large']
for llm in llms:
    if baseline:
        llm='onehot'


    if baseline:
            cui2idx1, embedding_matrix1=emb_dic(df1, baseline, None, df2)
            cui2idx2, embedding_matrix2=emb_dic(df2, baseline, None, df1)
        # embedding_cui= llm_embedding_2("/Users/yuhe/Desktop/AIPH/weekly/2023/2023.12/embeddings_project8/UMLS_code_openai.h5", "/Users/yuhe/Desktop/AIPH/weekly/2023/2023.12/embeddings_project8/UMLS_code2_openai.h5")
    else:
        llm_emb=f"cui_embeddings/concept_pair_{llm}.h5"
        embedding_cui=llm_embedding(llm_emb)
        cui2idx1, embedding_matrix1=emb_dic(df1,baseline, embedding_cui,df2)
        cui2idx2, embedding_matrix2=emb_dic(df2,baseline, embedding_cui,df1)


    # embedding_cui= llm_embedding_2("/Users/yuhe/Desktop/AIPH/weekly/2023/2023.12/embeddings_project8/UMLS_code_openai.h5", "/Users/yuhe/Desktop/AIPH/weekly/2023/2023.12/embeddings_project8/UMLS_code2_openai.h5")

    # print(embedding_matrix1.shape)
    # print(embedding_matrix2.shape)
    # print(cui2idx1)
    # print(cui2idx2)

    if baseline:
        inchannels=embedding_matrix1.shape[0]
    else:
        inchannels=len(embedding_cui.columns)

    source_train_loader, source_val_loader, target_train_loader, target_val_loader, target_test_loader, cui_channel=prepareSourceTargetDataset(df1,df2, disease, batch_size, cui2idx1, cui2idx2)
    print(cui_channel)

    # seeds=[215]
    # seeds=[401]
    seeds=[114514]
    seeds_n=''
    for i in seeds:
        seeds_n+="_"+str(i)

    stride=1
    padding=0

    result_name=f"best_results/small/TL_{llm}_bs{batch_size}_e{epoch}_seed{seeds_n}"

    print("Starting training model: ")
    classifier, log=model_running(device, source_train_loader, source_val_loader,embedding_matrix1, cui2idx1, seeds, epoch, in_channels=inchannels, stride=stride, padding=padding,filter_sizes=[2,3,4])
    # classifier=model_running(device, train_loader, val_loader,embedding_matrix, cui2idx, seed, epoch, in_channels=len(embedding_cui.columns), stride=stride, padding=padding, filter_sizes=[cui_channel])  ## only 1 filter with width of all channel
    # classifier=model_running(device, train_loader, embedding_matrix, cui2idx, seed, epoch, in_channels=cui_channel, stride=stride, padding=padding)

    print("Starting tuning on target: ")
    tune_classifier,log_tune=targetTune(device, classifier, target_train_loader, target_val_loader, embedding_matrix2, cui2idx2, seeds, epoch,in_channels=inchannels)

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
    r1=performance(device, tune_classifier, target_test_loader, result_name+"_test.csv")
    print("train performance:")
    r2=performance(device, tune_classifier, target_train_loader,result_name+"_train.csv")

    log_name=f"experimentLog/small/TL_experiment_log_{llm}_bs{batch_size}_e{epoch}_seed{seeds_n}.txt"

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

        f.write("\nStart Training Model:\n")
        f.write(log)
        f.write("Training Complete.\n")

        f.write("\nStart Tuning Model:\n")
        f.write(log_tune)
        f.write("Tuning Complete.\n")

        f.write("\n############## Model performance #############\n")
        f.write("test performance: \n")
        f.write(r1)
        f.write("train performance: \n")
        f.write(r2)
    f.close()

    if baseline:
        break