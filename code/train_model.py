import numpy as np
import logging
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import random
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import os
import time
import utilities as util
from tqdm.autonotebook import trange, tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
cache_dir = os.path.abspath('../models/gpt2_cache')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = GPT2Config.from_pretrained('gpt2', cache_dir=cache_dir)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=False, cache_dir=cache_dir)
trainset = util.TextDataset('trainingData.txt', tokenizer)
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', config=config, cache_dir=cache_dir)

gpt2.to(device)

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.WARN)
                        
util.train(trainset, gpt2, tokenizer, device, epochs=6, logger=logger)

prompt = '<|endoftext|>'
context_tokens = tokenizer.encode(prompt, add_special_tokens=False)
out = util.sample_sequence(gpt2, context_tokens, length=500, temperature=1, top_k=0, device=device)[:, 1:]

out = out[:, len(context_tokens):].tolist()

prefix = prompt
for o in out:
    text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
    text = text[:]
    print(prompt + text.replace('<|endoftext|>', '\n'))
time_str = time.strftime("%Y%m%d-%H%M%S")
txt_path = os.path.abspath('../outputs/')
with open(f'{txt_path}/text_{time_str}.txt', 'w') as f:
    f.write(prompt.replace('<|endoftext|>', '\n') + text.replace('<|endoftext|>', '\n'))

dir_name = f'GPT-2_{time_str}'
path = os.path.abspath('../models/')
os.mkdir(path + '/' + dir_name)
gpt2.save_pretrained(path + '/' + dir_name)
