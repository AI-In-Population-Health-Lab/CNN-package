# Transfer-Learning-with-embeddings-from-LLM

All codes are stored in `Clean_version` folder. Please cite this paper. Gao Y, Bao R, Ji Y, Sun Y, Song C, Ferraro JP, Ye Y. Transfer Learning with Clinical Concept Embeddings from Large Language Models. arXiv preprint arXiv:2409.13893. 2024 Sep 20. https://arxiv.org/pdf/2409.13893 

## Before Execution:

- **preprocess.py**: Extracts pre-trained embeddings from LLMs. The resulting embeddings are saved in the `cui_embeddings` folder.

- **cnn_feedforward.py**: Defines the CNN structure used for classification tasks.

- Please organize your data appopriately, and modify the **model_method.py** to integrate your data, if necessary.

## Model Execution:

- **runModel.py**: Runs the local CNN model.

- **runTL_notune.py**: Directly runs the shared model on the target site without any tuning.

- **runTL_withFreeze.py**: Runs the shared models with the convolutional layer frozen, tuning only the fully connected layer.

- **runTL.py**: Runs the shared models with both the convolutional and fully connected layers tuned.

## Output:

The results are returned as:
1. Best models in `.pth` format.
2. Experiment performance records in `.txt` format.
3. Classified case probabilities in `.csv` format.
