import torch
import transformers
from tqdm import tqdm
from typing import List
import tensorflow as tf
import math
import numpy as np 

# defining Dataset Class
class Dset(torch.utils.data.Dataset):
     """A custom dataset"""
     def __init__(self, data: list[list[int]]):
         self.data = []
         for d in data:
             input_ids = torch.tensor(d, dtype=torch.int64)
             attention_mask = torch.ones(len(d), dtype=torch.int64)
             self.data.append({'input_ids': input_ids,
                  'attention_mask': attention_mask, 'labels': input_ids})
 
     def __len__(self):
         return len(self.data)
 
     def __getitem__(self, idx: int):
         return self.data[idx]

def get_model_tokenizer(max_seq_len: int) -> tuple[transformers.GPT2LMHeadModel, transformers.GPT2Tokenizer]:
    """Gets a pretrained poetry model and GPT-2 tokenizer and setsg configurations 
     for the model and tokenizer.
    Args:
      max_seq_len (int): the maximum length of each tokenized datapoint      
    Returns:
      model: a GPT2 pretrained poetry model
      tokenizer: a GPT2 tokenizer
    """

    # set model configurations
    config = transformers.GPT2Config.from_pretrained('gpt2', from_tf=True)
    config.do_sample = True
    config.max_length = max_seq_len

    # get tokenizer
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("ashiqabdulkhader/GPT2-Poet", from_tf=True)

    # add padding token to tokenizer
    tokenizer.add_special_tokens({'pad_token': "[PAD]"})

    # set model configuration
    model = transformers.GPT2LMHeadModel.from_pretrained("ashiqabdulkhader/GPT2-Poet", config=config, from_tf=True)
    # add pad token to model's configuration
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def train_model(model: transformers.GPT2LMHeadModel, dset_train: Dset, dset_val: Dset, genre: str, save_model_filename: str, batch_size: int=4, epochs: int=1, lr: float=1e-3) -> transformers.GPT2LMHeadModel:
    """ Fine tunes a model using the given training and validation data and saves the model to the specified file name. Model is trainined
    with the given batch, epoch, and learning rate parameters. 
    Args:
      model (transformers.GPT2LMHeadModel): pretrained GPT2 model
      dset_train (Dset): training dataset
      dset_val (Dset): validation dataset
      genre (str): genre of the data
      save_model_filename (str): name of the file the model will be saved to
      batch_size (int): batch size to use when training
      epochs (int) : number of epochs to use during trianing
      lr (float) : learning rate to use when training

    Returns:
      model: the fine tuned model
    """
    
    # set training arguments
    training_args = transformers.TrainingArguments(
     output_dir="gpt2-poetry-model_save/training_args",
     learning_rate=lr,
     per_device_train_batch_size=batch_size, 
     per_device_eval_batch_size=batch_size, 
     num_train_epochs=epochs,
     evaluation_strategy='epoch',
     save_strategy='no',
    )

    # train model
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
    )
    trainer.train()  

    # save model
    file_path = "gpt2_trained_models/"+ genre.lower()+'/'+save_model_filename+"/"
    model.save_pretrained(file_path)  

    return model

def generate_texts(model: transformers.GPT2LMHeadModel, tokenizer: transformers.GPT2Tokenizer, n_texts: int, file_path=None) -> List[List[str]]:
    """ Generates texts from the given model and tokenizer and save the texts to a file path if a path is given.
    Args:
      model (transformers.GPT2LMHeadModel): pretrained GPT2 model
      tokenizer (Dset): training dataset
      n_texts (int): number of texts to generate

    Returns:
      texts: list of list of str generated texts that are in their tokenized forms
    """
    # intialize generated texts
    gen_texts = []
    # get special tokens that will be ignored in text generation
    special_tokens = [tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token, tokenizer.unk_token]  # Add any other special tokens you want to skip
    special_token_ids = [tokenizer.convert_tokens_to_ids(special_tokens)]

    # generate texts
    for _ in range(n_texts):
        # get encoded generated text
        encoded_output = model.generate(remove_invalid_values=True, max_length=10, do_sample=True, bad_words_ids=special_token_ids).numpy().tolist()[0]
        # decode the generated text
        text_output = tokenizer.batch_decode(encoded_output, skip_special_tokens=True)
        # remove special tokens from generated text
        text_output = [tok for tok in text_output if tok not in special_tokens]
        # add generated text to final results
        gen_texts.append(text_output)
    
    if file_path is not None:
        with open(file_path, 'w') as f:
          for text in gen_texts:
            f.write(' '.join(text)+ '\n')

    return gen_texts

def load_model(file_path):
    """ 
    Load a saved model

    Args:
      file_path (str): path to the saved gpt2 model

    Returns:
      the model associated with the file path
    """
    loaded_model = transformers.TFGPT2LMHeadModel.from_pretrained(file_path, from_pt=False)
    
    return loaded_model


def compute_perplexity(path_to_model: str, tokenizer: transformers.GPT2Tokenizer, test_data: List[str], max_length: int=10) -> float:
  """ Computes model perplexity on test data.
    Args:
      path_to_model: str file path to the saved fine-tuned GPT2 model
      tokenizer (transformers.GPT2Tokenizer): pretrained GPT2 tokenizer
      test_data (List of str): the test data in its tokenized form
      max_len (int): the maximum length of each tokenized datapoint

    Returns:
      ppl: float perplexity of test data
  """
  # load the saved model
  loaded_model = load_model(path_to_model)

  encodings = tokenizer("\n\n".join(test_data), return_tensors="tf")

  max_length = max_length
  stride=10
  seq_len = encodings.input_ids.shape[1]

  nlls = []
  prev_end_loc = 0
  for begin_loc in tqdm(range(0, seq_len, stride)):
      end_loc = min(begin_loc + max_length, seq_len)
      trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
      input_ids = encodings.input_ids[:, begin_loc:end_loc]
      target_ids_np = tf.identity(input_ids).numpy()
      target_ids_np[:, :-trg_len] = -100 
      target_ids = tf.convert_to_tensor(np.array(target_ids_np))
      with torch.no_grad():
          outputs = loaded_model(input_ids, labels=target_ids)

          # loss is calculated using CrossEntropyLoss which averages over valid labels
          # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
          # to the left by 1.
          neg_log_likelihood = outputs.loss

      nlls.append(neg_log_likelihood)

      prev_end_loc = end_loc
      if end_loc == seq_len:
          break
      
  ppl = math.exp(np.mean(nlls))
  return ppl
