from model_method import processDataset, llm_embedding, llm_embedding_2, emb_dic, splitTrainValTest, model_running, performance, changeOrder, prepareSourceTargetDataset,targetTune
import torch

disease="FLU"
batch_size=32
epoch=10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

disease_label="M to F"  #"delete M"      #"M to F"
cui_label= None    #"M to A"


df2= processDataset("/Users/yuhe/Desktop/AIPH/weekly/2024/March/cleaned_AC.xlsx", disease, disease_label, cui_label)
df1= processDataset("/Users/yuhe/Desktop/AIPH/weekly/2024/March/cleaned_SLC_87.xlsx", disease, disease_label, cui_label)
# df=changeOrder(df)
embedding_cui=llm_embedding("cui_embeddings/concept_pair_biobert.h5")

# embedding_cui= llm_embedding_2("/Users/yuhe/Desktop/AIPH/weekly/2023/2023.12/embeddings_project8/UMLS_code_openai.h5", "/Users/yuhe/Desktop/AIPH/weekly/2023/2023.12/embeddings_project8/UMLS_code2_openai.h5")
cui2idx1, embedding_matrix1=emb_dic(df1,embedding_cui,df2)
cui2idx2, embedding_matrix2=emb_dic(df2,embedding_cui,df1)

# print(embedding_matrix1.shape)
# print(embedding_matrix2.shape)
# print(cui2idx1)
# print(cui2idx2)


source_train_loader, source_val_loader, target_train_loader, target_val_loader, target_test_loader, cui_channel=prepareSourceTargetDataset(df1,df2, disease, batch_size, cui2idx1, cui2idx2)
print(cui_channel)

seeds=[215]

stride=1
padding=0

print("Starting training model: ")
classifier=model_running(device, source_train_loader, source_val_loader,embedding_matrix1, cui2idx1, seeds, epoch, in_channels=len(embedding_cui.columns), stride=stride, padding=padding,filter_sizes=[2,3,4])
# classifier=model_running(device, train_loader, val_loader,embedding_matrix, cui2idx, seed, epoch, in_channels=len(embedding_cui.columns), stride=stride, padding=padding, filter_sizes=[cui_channel])  ## only 1 filter with width of all channel
# classifier=model_running(device, train_loader, embedding_matrix, cui2idx, seed, epoch, in_channels=cui_channel, stride=stride, padding=padding)

print("Starting tuning on target: ")
tune_classifier=targetTune(device, classifier, target_train_loader, target_val_loader, embedding_matrix2, cui2idx2, seeds, epoch,in_channels=len(embedding_cui.columns))

print("############## Model Setting ###############")
print("Batch Size: ", batch_size)
print("Epoch: ", epoch)
print("Seed: ", seeds)
print("Stride: ", stride)
print("Padding: ", padding)
print("Disease_label:", disease_label)
print("CUI_label: ",cui_label)

print("############## Model performance #############")
print("test performance:")
performance(device, tune_classifier, target_test_loader)
print("train performance:")
performance(device, tune_classifier, target_train_loader)