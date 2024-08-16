from model_method import processDataset, llm_embedding, llm_embedding_2, emb_dic, splitTrainValTest, model_running, performance, changeOrder
import torch

disease="FLU"
batch_size=32
epoch=10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

disease_label="M to F"  #"delete M"      #"M to F"
cui_label= None    #"M to A"


df= processDataset("/Users/yuhe/Desktop/AIPH/weekly/2024/March/cleaned_AC.xlsx", disease, disease_label, cui_label)
# df=changeOrder(df)
embedding_cui=llm_embedding("cui_embeddings/concept_pair_openai_large.h5")
# embedding_cui= llm_embedding_2("/Users/yuhe/Desktop/AIPH/weekly/2023/2023.12/embeddings_project8/UMLS_code_openai.h5", "/Users/yuhe/Desktop/AIPH/weekly/2023/2023.12/embeddings_project8/UMLS_code2_openai.h5")
cui2idx, embedding_matrix=emb_dic(df,embedding_cui)
embedding_cui.to_excel("check.xlsx")

train_loader, val_loader, test_loader, cui_channel=splitTrainValTest(df, disease, batch_size, cui2idx)
print(cui_channel)

seeds=[215]

stride=1
padding=0

print("Starting training model: ")
classifier=model_running(device, train_loader, val_loader,embedding_matrix, cui2idx, seeds, epoch, in_channels=len(embedding_cui.columns), stride=stride, padding=padding,filter_sizes=[2,3,4])
# classifier=model_running(device, train_loader, val_loader,embedding_matrix, cui2idx, seed, epoch, in_channels=len(embedding_cui.columns), stride=stride, padding=padding, filter_sizes=[cui_channel])  ## only 1 filter with width of all channel
# classifier=model_running(device, train_loader, embedding_matrix, cui2idx, seed, epoch, in_channels=cui_channel, stride=stride, padding=padding)



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
performance(device, classifier, test_loader)
print("train performance:")
performance(device, classifier, train_loader)