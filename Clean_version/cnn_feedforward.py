import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN_feedforward(nn.Module):
    def __init__(self, pretrained_embedding=None,
                 freeze_embedding=True,
                 cuis_size=None,
                 embed_dim=768,
                 filter_sizes=[1],
                 num_filters=[100],
                 num_classes=2,
                 dropout=0.5,
                 in_channels=768,
                 stride=0,
                 padding=0,
                 freezeCL=False):
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
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)
        # # self.conv1 = nn.Conv2d(number_of_channel, number_of_classes, kernel_size, stride, padding)
        # self.conv1 = nn.ModuleList([
        #     nn.Conv2d(1, number_of_channel, (fs, embedding_dim)) for fs in filter_sizes
        # ])
        # self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        if freezeCL:
            # for param in self.embedding.parameters():
            #     param.requires_grad = False  # Freeze the embedding layer

            for conv1d in self.conv1d_list:
                for param in conv1d.parameters():
                    param.requires_grad = False  # Freeze all convolutional layers

            # Only the fully connected layer will be trainable
            for param in self.fc.parameters():
                param.requires_grad = True


    def forward(self, input_ids, mask=False):
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
        # print('x_embed1: ', x_embed.shape()) #32*65*768

        if mask:
            mask = (input_ids != 0).long()
            mask = mask.unsqueeze(-1).float()  # Shape: (b, max_len, 1)
            x_embed = x_embed * mask
            # print('x_embed2: ', x_embed) #32*65*768

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)
        # print(x_reshaped.shape)
        #32*768*65

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
        # print(self.conv1d_list[0](x_reshaped).shape)
        # conv1d[0] 32*100*64

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        # print(x_conv_list[0].shape[2])
        # print(x_pool_list[0].shape) # 32* 100 *1
        
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],dim=1)
        # print(x_fc.shape) # 32*300
        
        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits