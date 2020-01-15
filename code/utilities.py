import numpy as np
import logging
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import random
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from tqdm.autonotebook import trange, tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter


class TextDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.examples = list()
        txt = str()
        with open(path, encoding='utf-8') as f:
            txt = f.read()
        
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(txt))
        
        for i in range(0, len(tokenized_text) - 512 + 1, 512):
            self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i + 512]))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        return torch.tensor(self.examples[item])
        
        
def train(train_dataset, model, tokenizer, device, sample_context='<|endoftext|>', batch_size=4, epochs=8, learning_rate=.0001, logger=None):
    c_tokens = tokenizer.encode(sample_context, add_special_tokens=False)
    tb_writer = SummaryWriter()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    t_total = len(train_dataloader)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_params = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
                                 'weight_decay': 0.0},
                                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
                                 'weight_decay': 0.0}
                               ]
    optimizer = AdamW(optimizer_grouped_params, lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", epochs)
    logger.info("  Instantaneous batch size per GPU = %d", batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   batch_size * 1 * (1))
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    
    tr_loss, logging_loss = 0.0, 0.0
    
    model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    train_iterator = trange(epochs_trained, epochs, desc='Epoch')
    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        for step, batch in enumerate(epoch_iterator):
            
            inputs, labels = (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.train()
            
            outputs = model(inputs, labels=labels)
            
            loss = outputs[0]
            
            loss.backward()
            
            tr_loss += loss.item()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            
            if global_step % 50 == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/50, global_step)
                    logging_loss = tr_loss
        
        demo = sample_sequence(model, c_tokens, length=200, temperature=1, top_k=0, device=device)
        demo = demo[:, len(c_tokens):].tolist()
        for d in demo:
            demo_text = tokenizer.decode(d, clean_up_tokenization_spaces=True)
            demo_text = demo_text[:]
        print(sample_context.replace('<|endoftext|>', '\n') + demo_text.replace('<|endoftext|>', '\n'))
    tb_writer.close()
    return global_step, tr_loss / global_step
    
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, source=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits
    
def sample_sequence(model, context, length=20, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated
