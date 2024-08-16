from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


extracted_cui = "/0814_seed-1494714102_size10000_I.csv"


extracted_df = pd.read_csv(extracted_cui)
wiki_path = "/Interested_CUI_wiki.csv"
wiki_df = pd.read_csv(wiki_path)
cuis = wiki_df.CUI.values.tolist()
titles = wiki_df.Title.values.tolist()
cui_dict = {}

for idx, (c, t) in enumerate(zip(cuis, titles)):
    if type(t) is float:
        print(idx)
    cui_dict[c] = {"pos":t,"neg":"Not " + t}

# NegBERT option replace the name bvanaken/clinical-assertion-negation-bert
#
tokenizer = AutoTokenizer.from_pretrained("bvanaken/clinical-assertion-negation-bert")
model = AutoModel.from_pretrained("bvanaken/clinical-assertion-negation-bert")


def get_embedding(text):
    """Get BERT embedding for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# store every cui and embeddings
new_cui_dict = {}
for cui, item in cui_dict.items():
    new_cui_dict[cui] = {"pos": get_embedding(item["pos"]), "neg": get_embedding(item["neg"])} # 1, 768
    # print(new_cui_dict[cui]["pos"].shape)
def convert_row_to_embedding(row, tokenizer=None, model=None):
    embeddings = []
    # print("row in convert_row_to_embedding", row) # 41
    for cui, value in row.items():
        if value == "P":
            embedding = new_cui_dict[cui]["pos"]
        else:
            embedding = new_cui_dict[cui]["neg"]
        embeddings.append(embedding)
    # # Combine all embeddings for the row
    combined_embedding = np.concatenate(embeddings, axis=0)
    return combined_embedding

embeddings_list = []
labels_list = []
for idx, row in extracted_df.iterrows():
    embedding = convert_row_to_embedding(row.drop(labels=[extracted_df.columns[-1]]), tokenizer, model)
    # label = row[extracted_df.columns[-1]]
    label = torch.tensor([row[extracted_df.columns[-1]]], dtype=torch.long)  # Convert label to tensor
    embeddings_list.append(embedding)
    labels_list.append(label)
# Convert embeddings_list and labels_list to tensors
embeddings_tensor = torch.tensor(embeddings_list)
labels_tensor = torch.cat(labels_list)
print(embeddings_tensor.shape)
print(labels_tensor.shape)
dataset = TensorDataset(embeddings_tensor, labels_tensor)


# CNN Model definition
class CNN_NLP(nn.Module):
    def __init__(self, embed_dim=768, filter_sizes=[3, 4, 5], num_filters=[100, 100, 100], num_classes=2, dropout=0.5):
        super(CNN_NLP, self).__init__()
        self.embedding = nn.Identity()
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters[i], kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, embeddings):
        x_reshaped = embeddings.permute(0, 2, 1)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
        logits = self.fc(self.dropout(x_fc))
        return logits

# Instantiate the CNN model and forward pass
# cnn_model = CNN_NLP()
# logits = cnn_model(cls_embeddings)


# 1. Data Preparation
# data_tensor = torch.randn(1000, 42, 768)  # Random tensor as placeholder. Replace with your data.
# labels_tensor = torch.randint(0, 2, (1000,)) # torch.Size([1000])



# 2. Split the data into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 3. Model definition (Using the CNN_NLP class you provided)
model = CNN_NLP()

# 4. Training
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        data, labels = batch
        optimizer.zero_grad()
        outputs = model(data)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

# 5. Evaluation
model.eval()
correct = 0
total = 0
labels_all = []
predictions_all = []

with torch.no_grad():
    for batch in test_loader:
        data, labels = batch
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        labels_all.extend(labels.tolist())
        predictions_all.extend(predicted.tolist())


tn, fp, fn, tp = confusion_matrix(labels_all, predictions_all).ravel()


precision = precision_score(labels_all, predictions_all)
recall = recall_score(labels_all, predictions_all)
f1 = f1_score(labels_all, predictions_all)
fpr = fp / (fp + tn)

print(f'Accuracy: {correct / total:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'False Positive Rate (FPR): {fpr:.2f}')


