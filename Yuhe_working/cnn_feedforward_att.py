import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class CNN_feedforward(nn.Module):
    def __init__(self, pretrained_embedding=None,
                 freeze_embedding=False, #was True
                 cuis_size=None,
                 embed_dim=None,
                 filter_sizes=None,
                 num_filters=[20],  # 20 filters, each has 1 out_channel
                 num_classes=2,
                 dropout=0.5,
                 in_channels=768,
                 stride=0,
                 padding=0):
        super(CNN_feedforward, self).__init__()
        # Embedding layer
        if pretrained_embedding is not None:
            self.cuis_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=cuis_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)


        # number_of_channel = symptom_size
        kernel_size = 3  # Size of the convolutional kernel
        # stride = 1  # Stride for the convolution
        # padding = 1  # Padding for the input
       
       # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,   #in_channels=self.embed_dim,
                      out_channels=num_filters[i], #out_channels= number of kernel (filter), which is randomly selected
                      kernel_size=filter_sizes[i],
                      stride=stride,
                      padding=padding
                      )
            for i in range(len(num_filters))
        ])
        self.sa1 = SelfAttention(20)
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)
        # # self.conv1 = nn.Conv2d(number_of_channel, number_of_classes, kernel_size, stride, padding)
        # self.conv1 = nn.ModuleList([
        #     nn.Conv2d(1, number_of_channel, (fs, embedding_dim)) for fs in filter_sizes
        # ])
        # self.pool = nn.MaxPool2d(kernel_size, stride, padding)


    def forward(self, input_ids):
        # print("x dtype")
        # print(x.dtype)
        # x = x.long()
        # print(x.dtype)
        # print(x.size())
        # x = self.embedding(x)
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # # x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # x = self.fc3(x)

        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim) 
        x_embed = self.embedding(input_ids.long()).float()
        # print("x_embed: ", x_embed.shape) #32*71*768

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)

        x_reshaped = x_embed.permute(0, 2, 1)
        # # print(x_reshaped.shape)
        # #32*768*71


        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list] 
        # print("conv1d: ", self.conv1d_list[0](x_embed).shape)
        # conv1d[0] 32*1*1

        x_att=[self.sa1(x for x in x_conv_list)]
        ############## no need to maxpool in this setting ####################
        # # Max pooling. Output shape: (b, num_filters[i], 1)
        # x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
        #     for x_conv in x_conv_list]
        # # print(x_conv_list[0].shape[2])
        # # print(x_pool_list[0].shape) # 32* 1 *1
        ######################################################################
        
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_att],
                         dim=1)
        # print(x_fc.shape) # 32*20
        
        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits