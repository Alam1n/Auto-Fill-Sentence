# Auto Fill Sentece
Auto-Fill Sentence is a machine learning project that uses the GPT-2 model to complete sentences based on a dataset of wikipedia article sentences. 

This repository contains the code and configuration for training a GPT-2 model to perform sentence completion.
It also contains the code for the web app but i can not host it due to the large model size.

The model is not perfect since i only used part of the dataset to reduce processing time. 

### Key Features
* Model: GPT-2
* Dataset: A text dataset used to train the model for sentence completion.
* Training Details:
* Epochs: 3
* Batch Size: 4
* Output Directory: ./results
* Save Steps: Every 10,000 steps
* Save Total Limit: 2
* Model Architecture
* The model utilizes the GPT-2 architecture, which is a Transformer-based model designed for natural language understanding and generation. The key components include:

* Tokenizer: GPT2Tokenizer is used to convert text into tokens that the GPT-2 model can process.
* Model: GPT2LMHeadModel is the language model head of GPT-2, designed for text generation tasks. It predicts the next word in a sequence given the preceding words.

### Training Setup
* Dataset Loading: The TextDataset class is used to load and preprocess the text data, with a block size of 128 tokens.
* Data Collator: DataCollatorForLanguageModeling is used to handle the data batching for language modeling tasks, with masked language modeling (MLM) disabled.
* Training Arguments: Configured with parameters such as the number of epochs, batch size, and output directory.
* Trainer: Trainer class from Hugging Face's Transformers library is used to handle the training process, including model saving and logging.

This setup trains the GPT-2 model to predict and complete sentences, leveraging the pre-trained capabilities of GPT-2 for text generation.
