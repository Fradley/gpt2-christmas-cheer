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
import argparse
import utilities as util
from tqdm.autonotebook import trange, tqdm
try:
	from torch.utils.tensorboard import SummaryWriter
except:
	from tensorboardX import SummaryWriter

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', default=4, type=int, help='Chooses the training batch size.')
	parser.add_argument('--num_epochs', default=4, type=int, help='Number of training epochs')
	parser.add_argument('--learning_rate', default=1e-4, type=float, help='Determines the learning rate of training.')
	parser.add_argument('--model_dir', default='gpt2', type=str, help='Uses a previous model')
	parser.add_argument('--save_model', default=False, type=bool, help='Whether to save model after training.')
	parser.add_argument('--generate_sample', default=False, type=bool, help='Determine whether to generate samples after training.')
	parser.add_argument('--sample_length', default=500, type=int, help='Number of tokens to generate in sample.')
	parser.add_argument('--sample_temperature', default=1, type=float, help='Sample generation temperature. Higer values indicate more "unlikely" words will be generated.')
	parser.add_argument('--sample_k', default=0, type=int, help='Tosses out words not a probabilistic as the top k words.')
	parser.add_argument('--prompt', default='<|endoftext|>', type=str, help='Initial text to base output on.')
	args = parser.parse_args()



	cache_dir = os.path.abspath('../models/gpt2_cache')
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	config = GPT2Config.from_pretrained('gpt2', cache_dir=cache_dir)
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=False, cache_dir=cache_dir)
	trainset = util.TextDataset('trainingData.txt', tokenizer)
	gpt2 = GPT2LMHeadModel.from_pretrained(args.model_dir, config=config, cache_dir=cache_dir)

	gpt2.to(device)

	logger = logging.getLogger(__name__)
	logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
			    datefmt = '%m/%d/%Y %H:%M:%S',
			    level = logging.WARN)

	util.train(trainset, gpt2, tokenizer, device,batch_size=args.batch_size, epochs=args.num_epochs, learning_rate=args.learning_rate, logger=logger)
	if args.generate_sample:
		params = vars(args)
		param_list = str(params) + '\n'
		prompt = args.prompt
		context_tokens = tokenizer.encode(prompt, add_special_tokens=False)
		out = util.sample_sequence(gpt2, context_tokens, length=args.sample_length, temperature=args.sample_temperature, top_k=args.sample_k, device=device)[:, 1:]

		out = out[:, len(context_tokens):].tolist()

		prefix = prompt
		for o in out:
			text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
			text = text[:]

		print(prompt + text.replace('<|endoftext|>', '\n'))
		time_str = time.strftime("%Y%m%d-%H%M%S")
		txt_path = os.path.abspath('../outputs/')
		with open(f'{txt_path}/text_{time_str}.txt', 'w') as f:
			f.write(param_list)
			f.write(prompt.replace('<|endoftext|>', '\n') + text.replace('<|endoftext|>', '\n'))
	if args.save_model:
		dir_name = f'GPT-2_{time_str}'
		path = os.path.abspath('../models/')
		os.mkdir(path + '/' + dir_name)
		gpt2.save_pretrained(path + '/' + dir_name)

if __name__ == '__main__':
	main()

