import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BasicTokenizer
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2DoubleHeadsModel
import csv
import numpy as np
import collections
import sys

import time
tokenized_input = []
num_entities = int(sys.argv[1])  #2
modelfolder = sys.argv[2] #'BertBase'

dir = 'data/'


def prepareData(infile):
  f = open(infile)
  inputlist_sent1 = []
  inputlist_sent2 = []
  tgtlist = []
  attractorlist_1 = []
  attractorlist_2 = []
  attractorlist_3 = []
  attractorlist_4 = []
  attractorlist_5 = []
  attractorlist_6 = []
  reader = csv.DictReader(f, delimiter='\t')
  for row in reader:

    inputlist_sent1.append(row['context'])
    inputlist_sent2.append(row['ending'])
    tgtlist.append(row['target_occupation'])
    attractorlist_1.append(row['attractor_1'])
    if num_entities == 2:
        continue
    attractorlist_2.append(row['attractor_2'])
    if num_entities == 3:
        continue
    attractorlist_3.append(row['attractor_3'])
    if num_entities == 4:
        continue
    attractorlist_4.append(row['attractor_4'])
    if num_entities == 5:
        continue
    attractorlist_5.append(row['attractor_5'])
    if num_entities == 6:
        continue
    attractorlist_6.append(row['attractor_6'])
    
  
  if num_entities == 6:
    return inputlist_sent1,inputlist_sent2,tgtlist,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4,attractorlist_5
  if num_entities == 5:
    return inputlist_sent1,inputlist_sent2,tgtlist,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4
  if num_entities == 4:
    return inputlist_sent1,inputlist_sent2,tgtlist,attractorlist_1,attractorlist_2,attractorlist_3
  if num_entities == 3:
    return inputlist_sent1,inputlist_sent2,tgtlist,attractorlist_1,attractorlist_2
  if num_entities == 2:
    return inputlist_sent1,inputlist_sent2,tgtlist,attractorlist_1
  return inputlist_sent1,inputlist_sent2,tgtlist,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4,attractorlist_5,attractorlist_6



def prep_input(inputlist_sent1,inputlist_sent2, tokenizer):
    for i in range(0, len(inputlist_sent1)):
        masked_index = None
        text = []
        #masking token for bert
        if bert:
            mtok = '[MASK]'
        #masking token for roberta
        if roberta:
            mtok = '<mask>'
        #classification token for bert
        if bert:
            text.append('[CLS]')
        #classification token for roberta
        if roberta:
            text.append('<s>')
        if gpt2:
            text.append('<|endoftext|>')
        text += inputlist_sent1[i].strip().split()

        text += inputlist_sent2[i].strip().split()

        if bert:
            text.append('[SEP]')
        #separator for roberta
        if roberta:
            text.append('</s>')
        text = ' '.join(text)

        tokenized_text = tokenizer.tokenize(text)
        tokenized_input.append(tokenized_text)
        if i == 0:
            print(tokenized_text)
        #input()
        if gpt2!=True:
            for index,tok in enumerate(tokenized_text):
                if tok == mtok: masked_index = index
        else:
            masked_index = -1 #for gpt2 predict the last word
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        
        tokens_tensor = torch.tensor([indexed_tokens])
        yield tokens_tensor, masked_index,tokenized_text

def get_predictions(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4,attractorlist_5,\
    attractorlist_6,tgtlist,model,tokenizer,k=5):
#def get_predictions(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4,attractorlist_5,\
#   tgtlist,model,tokenizer,k=5):
#def get_predictions(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4,\
#  tgtlist,model,tokenizer,k=5):
#def get_predictions(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,attractorlist_3,\
#    tgtlist,model,tokenizer,k=5):
#def get_predictions(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,\
# tgtlist,model,tokenizer,k=5):
#def get_predictions(inputlist_sent1,inputlist_sent2,attractorlist_1,\
#    tgtlist,model,tokenizer,k=5):
    all_distractor = ['flowers','glasses','meat','bread','fish','paintings','furniture']
    all_distractor_prob = []
    token_preds = []
    tok_probs = []
    token_probs = []
    attractor_1_prob = []
    attractor_2_prob = []
    attractor_3_prob = []
    attractor_4_prob = []
    attractor_5_prob = []
    attractor_6_prob = []
    count = 0
    start_time = time.time()
    for i,(tokens_tensor, masked_word_index,_) in enumerate(prep_input(inputlist_sent1,inputlist_sent2,tokenizer)):

        with torch.no_grad():
            #works for bert and roberta
            predictions = model(tokens_tensor)[0]

        predicted_tokens = []
        predicted_token_probs = []
        tgt = tgtlist[i]  #target for data instance i
        attractor_1 = attractorlist_1[i]
        attractor_2 = attractorlist_2[i]
        attractor_3 = attractorlist_3[i]
        attractor_4 = attractorlist_4[i]
        attractor_5 = attractorlist_5[i]
        attractor_6 = attractorlist_6[i]
        #for roberta adding extra symbol
        if roberta or gpt2:
            tgt = '\u0120'+tgt
            attractor_1 = '\u0120'+attractor_1
            attractor_2 = '\u0120'+attractor_2
            attractor_3 = '\u0120'+attractor_3
            attractor_4 = '\u0120'+attractor_4
            attractor_5 = '\u0120'+attractor_5
            attractor_6 = '\u0120'+attractor_6

        softpred = torch.softmax(predictions[0,masked_word_index],0)

        top_inds = torch.argsort(softpred,descending=True)[:k].cpu().numpy()
        
        top_probs = [softpred[tgt_ind].item() for tgt_ind in top_inds]
        top_tok_preds = tokenizer.convert_ids_to_tokens(top_inds)
        token_preds.append(top_tok_preds)
        tok_probs.append(top_probs)

        try:
            tgt_ind = tokenizer.convert_tokens_to_ids([tgt])[0]
            attractor_ind_1 = tokenizer.convert_tokens_to_ids([attractor_1])[0]
            attractor_ind_2 = tokenizer.convert_tokens_to_ids([attractor_2])[0]
            attractor_ind_3 = tokenizer.convert_tokens_to_ids([attractor_3])[0]
            attractor_ind_4 = tokenizer.convert_tokens_to_ids([attractor_4])[0]
            attractor_ind_5 = tokenizer.convert_tokens_to_ids([attractor_5])[0]
            attractor_ind_6 = tokenizer.convert_tokens_to_ids([attractor_6])[0]
            if roberta or gpt2:
                this_all_distractor_ind = [tokenizer.convert_tokens_to_ids(['\u0120'+i])[0] for i in all_distractor]
            else:
                this_all_distractor_ind = [tokenizer.convert_tokens_to_ids([i])[0] for i in all_distractor]

        except:
            print('not found id for this token in vocabulary',tgt)
            this_tgt_prob = np.nan
            this_attractor_1_prob = np.nan
        else:
            this_tgt_prob = softpred[tgt_ind].item()
            this_attractor_1_prob = softpred[attractor_ind_1].item()
            this_attractor_2_prob = softpred[attractor_ind_2].item()
            this_attractor_3_prob = softpred[attractor_ind_3].item()
            this_attractor_4_prob = softpred[attractor_ind_4].item()
            this_attractor_5_prob = softpred[attractor_ind_5].item()
            this_attractor_6_prob = softpred[attractor_ind_6].item()

            this_all_distractor_prob = [softpred[j].item() for j in this_all_distractor_ind]
            all_distractor = ['flowers','glasses','meat','bread','fish','paintings','furniture']
            all_distractor_prob_dict = {}
            all_distractor_prob_dict['flowers'] = this_all_distractor_prob[0]
            all_distractor_prob_dict['glasses'] = this_all_distractor_prob[1]
            all_distractor_prob_dict['meat'] = this_all_distractor_prob[2]
            all_distractor_prob_dict['bread'] = this_all_distractor_prob[3]
            all_distractor_prob_dict['fish'] = this_all_distractor_prob[4]
            all_distractor_prob_dict['paintings'] = this_all_distractor_prob[5]
            all_distractor_prob_dict['furniture'] = this_all_distractor_prob[6]

        token_probs.append(this_tgt_prob)
        attractor_1_prob.append(this_attractor_1_prob)
        attractor_2_prob.append(this_attractor_2_prob)
        attractor_3_prob.append(this_attractor_3_prob)
        attractor_4_prob.append(this_attractor_4_prob)
        attractor_5_prob.append(this_attractor_5_prob)
        attractor_6_prob.append(this_attractor_6_prob)
        all_distractor_prob.append(all_distractor_prob_dict)
        if count%200==0:
            print('completed lines ',count,' in time ',(time.time()-start_time))
            start_time = time.time()
        count+=1
    return token_preds,tok_probs,token_probs,attractor_1_prob,attractor_2_prob,attractor_3_prob,attractor_4_prob,attractor_5_prob,attractor_6_prob,all_distractor_prob
    #return token_preds,tok_probs,token_probs,attractor_1_prob,attractor_2_prob,attractor_3_prob,attractor_4_prob,attractor_5_prob,all_distractor_prob
    #return token_preds,tok_probs,token_probs,attractor_1_prob,attractor_2_prob,attractor_3_prob,attractor_4_prob,all_distractor_prob
    #return token_preds,tok_probs,token_probs,attractor_1_prob,attractor_2_prob,attractor_3_prob,all_distractor_prob
    #return token_preds,tok_probs,token_probs,attractor_1_prob,attractor_2_prob,all_distractor_prob
    #return token_preds,tok_probs,token_probs,attractor_1_prob,all_distractor_prob




def load_model(modeldir):
    # Load pre-trained model tokenizer (vocabulary)
    #it will download the vocab.txt
    if roberta:
        tokenizer = RobertaTokenizer.from_pretrained(modeldir)
        model = RobertaForMaskedLM.from_pretrained(modeldir)
    if gpt2:
        tokenizer = GPT2Tokenizer.from_pretrained(modeldir)
        model = GPT2DoubleHeadsModel.from_pretrained(modeldir)
    if bert:
        tokenizer = BertTokenizer.from_pretrained(modeldir)
        model = BertForMaskedLM.from_pretrained(modeldir)
    
    model.eval()
    return model,tokenizer

klist = [1,10]
#modeldir = 'bert-large-uncased'

#modeldir = 'bert-base-uncased'
#modeldir = 'roberta-large'
#modeldir = 'roberta-base'
#modeldir = 'gpt2-large'
#modeldir = 'gpt2-medium'
#modeldir = 'gpt2-xl'
modeldir = 'gpt2'
gpt2 = True
roberta=False
bert=False

bert_base,tokenizer_base = load_model(modeldir)


def test_fk_acc(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4,attractorlist_5,attractorlist_6,tgtlist,model,tokenizer,k=5):
#def test_fk_acc(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4,attractorlist_5,tgtlist,model,tokenizer,k=5):
#def test_fk_acc(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4,tgtlist,model,tokenizer,k=5):
#def test_fk_acc(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,attractorlist_3,tgtlist,model,tokenizer,k=5):
#def test_fk_acc(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,tgtlist,model,tokenizer,k=5):
#def test_fk_acc(inputlist_sent1,inputlist_sent2,attractorlist_1,tgtlist,model,tokenizer,k=5):

    start = time.time()
    if num_entities == 7:
        tok_preds,top_probs,target_probs, attractor_1_probs,attractor_2_probs,attractor_3_probs,attractor_4_probs,attractor_5_probs,\
    attractor_6_probs, all_distractor_prob = get_predictions(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4,attractorlist_5,\
    attractorlist_6,tgtlist,model,tokenizer,k)
    if num_entities == 6:
        tok_preds,top_probs,target_probs, attractor_1_probs,attractor_2_probs,attractor_3_probs,attractor_4_probs,\
    attractor_5_probs, all_distractor_prob = get_predictions(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,attractorlist_3,\
    attractorlist_4,attractorlist_5, tgtlist,model,tokenizer,k)
    if num_entities == 5:
        tok_preds,top_probs,target_probs, attractor_1_probs,attractor_2_probs,attractor_3_probs,attractor_4_probs, all_distractor_prob = \
    get_predictions(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,attractorlist_3,\
    attractorlist_4,tgtlist,model,tokenizer,k)
    if num_entities == 4:
        tok_preds,top_probs,target_probs, attractor_1_probs,attractor_2_probs,attractor_3_probs, all_distractor_prob = \
    get_predictions(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,attractorlist_3,\
    tgtlist,model,tokenizer,k)
    if num_entities == 3:
        tok_preds,top_probs,target_probs, attractor_1_probs,attractor_2_probs, all_distractor_prob = \
    get_predictions(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,\
    tgtlist,model,tokenizer,k)
    if num_entities == 2:
        tok_preds,top_probs,target_probs, attractor_1_probs, all_distractor_prob = \
    get_predictions(inputlist_sent1,inputlist_sent2,attractorlist_1, \
    tgtlist,model,tokenizer,k)
    
    if num_entities == 7:
        fieldnames = ['context','ending','target_occupation','attractor_1','attractor_2','attractor_3','attractor_4',\
    'attractor_5','attractor_6','target_prob','attractor_1_prob','attractor_2_prob','attractor_3_prob','attractor_4_prob',\
    'attractor_5_prob','attractor_6_prob','ordered_items','model_favoured_words','model_favoured_probability']
    if num_entities == 6:
        fieldnames = ['context','ending','target_occupation','attractor_1','attractor_2','attractor_3','attractor_4',\
    'attractor_5','target_prob','attractor_1_prob','attractor_2_prob','attractor_3_prob','attractor_4_prob',\
    'attractor_5_prob','ordered_items','model_favoured_words','model_favoured_probability']
    if num_entities == 5:
        fieldnames = ['context','ending','target_occupation','attractor_1','attractor_2','attractor_3','attractor_4',\
    'target_prob','attractor_1_prob','attractor_2_prob','attractor_3_prob','attractor_4_prob','ordered_items',\
    'model_favoured_words','model_favoured_probability']
    if num_entities == 4:
        fieldnames = ['context','ending','target_occupation','attractor_1','attractor_2','attractor_3',\
    'target_prob','attractor_1_prob','attractor_2_prob','attractor_3_prob','ordered_items',\
    'model_favoured_words','model_favoured_probability']
    if num_entities == 3:
        fieldnames = ['context','ending','target_occupation','attractor_1','attractor_2',\
    'target_prob','attractor_1_prob','attractor_2_prob','ordered_items',\
    'model_favoured_words','model_favoured_probability']
    if num_entities == 2:
        fieldnames = ['context','ending','target_occupation','attractor_1',\
    'target_prob','attractor_1_prob', 'ordered_items',\
    'model_favoured_words','model_favoured_probability']

    top_k_score = 0
    probability_plausible_file = open(dir+'job_training_introducing_single_entity_with_'+str(num_entities-1)+'_distractorResults'+modelfolder+'.csv','w')
    #probability_plausible_file = open(dir+'job_training_introducing_'+str(num_entities)+'_entitiesResults'+modelfolder+'.csv','w')
    writer = csv.DictWriter(probability_plausible_file, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    for i in range(0, len(tok_preds)):
        candidate = tok_preds[i]

        if tgtlist[i].lower() in [x.lower() for x in candidate]:
            response = 'correct'
            top_k_score+=1
        else:
            response = 'incorrect'
        count = 0
        try:
            value = tgtlist[i]
        except:
            print('not found', tgtlist[i])
            value = 'not found'
        finally:
            itemlist = {}
            print(all_distractor_prob[i])
            itemlist = all_distractor_prob[i]
            ordered_items = [k for k,v in sorted(itemlist.items(), key=lambda item:item[1], reverse=True)]
            if num_entities == 7:
                writer.writerow({'context':inputlist_sent1[i],'ending':inputlist_sent2[i],'target_occupation':tgtlist[i],\
                'attractor_1':attractorlist_1[i],'attractor_2':attractorlist_2[i],'attractor_3':attractorlist_3[i],\
                'attractor_4':attractorlist_4[i],\
                'attractor_5':attractorlist_5[i], \
                'attractor_6':attractorlist_6[i],
                'target_prob':target_probs[i],\
                'attractor_1_prob':attractor_1_probs[i],'attractor_2_prob':attractor_2_probs[i],\
                'attractor_3_prob':attractor_3_probs[i],'attractor_4_prob':attractor_4_probs[i],\
                'attractor_5_prob':attractor_5_probs[i],\
                'attractor_6_prob':attractor_6_probs[i],\
                'model_favoured_words':tok_preds[i],\
                'model_favoured_probability':top_probs[i],
                'ordered_items':ordered_items})
            if num_entities == 6:
                writer.writerow({'context':inputlist_sent1[i],'ending':inputlist_sent2[i],'target_occupation':tgtlist[i],\
                'attractor_1':attractorlist_1[i],'attractor_2':attractorlist_2[i],'attractor_3':attractorlist_3[i],\
                'attractor_4':attractorlist_4[i],\
                'attractor_5':attractorlist_5[i], 'target_prob':target_probs[i],\
                'attractor_1_prob':attractor_1_probs[i],'attractor_2_prob':attractor_2_probs[i],\
                'attractor_3_prob':attractor_3_probs[i],'attractor_4_prob':attractor_4_probs[i],\
                'attractor_5_prob':attractor_5_probs[i],\
                'model_favoured_words':tok_preds[i],\
                'model_favoured_probability':top_probs[i],
                'ordered_items':ordered_items})
            if num_entities == 5:
                writer.writerow({'context':inputlist_sent1[i],'ending':inputlist_sent2[i],'target_occupation':tgtlist[i],\
                'attractor_1':attractorlist_1[i],'attractor_2':attractorlist_2[i],'attractor_3':attractorlist_3[i],'attractor_4':attractorlist_4[i],\
                'target_prob':target_probs[i],\
                'attractor_1_prob':attractor_1_probs[i],'attractor_2_prob':attractor_2_probs[i],\
                'attractor_3_prob':attractor_3_probs[i],'attractor_4_prob':attractor_4_probs[i],\
                'model_favoured_words':tok_preds[i],\
                'model_favoured_probability':top_probs[i],
                'ordered_items':ordered_items})
            if num_entities == 4:
                writer.writerow({'context':inputlist_sent1[i],'ending':inputlist_sent2[i],'target_occupation':tgtlist[i],\
                'attractor_1':attractorlist_1[i],'attractor_2':attractorlist_2[i],'attractor_3':attractorlist_3[i],\
                'target_prob':target_probs[i],\
                'attractor_1_prob':attractor_1_probs[i],'attractor_2_prob':attractor_2_probs[i],\
                'attractor_3_prob':attractor_3_probs[i],\
                'model_favoured_words':tok_preds[i],\
                'model_favoured_probability':top_probs[i],
                'ordered_items':ordered_items})
            if num_entities == 3:
                writer.writerow({'context':inputlist_sent1[i],'ending':inputlist_sent2[i],'target_occupation':tgtlist[i],\
                'attractor_1':attractorlist_1[i],'attractor_2':attractorlist_2[i],\
                'target_prob':target_probs[i],\
                'attractor_1_prob':attractor_1_probs[i],'attractor_2_prob':attractor_2_probs[i],\
                'model_favoured_words':tok_preds[i],\
                'model_favoured_probability':top_probs[i],
                'ordered_items':ordered_items})
            if num_entities == 2:
                writer.writerow({'context':inputlist_sent1[i],'ending':inputlist_sent2[i],'target_occupation':tgtlist[i],\
                'attractor_1':attractorlist_1[i],\
                'target_prob':target_probs[i],\
                'attractor_1_prob':attractor_1_probs[i],\
                'model_favoured_words':tok_preds[i],\
                'model_favoured_probability':top_probs[i],
                'ordered_items':ordered_items})
    

start = time.time()
if num_entities == 7:
    inputlist_sent1,inputlist_sent2,tgtlist,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4,attractorlist_5,\
attractorlist_6 = prepareData(dir+'job_training_introducing_single_entity_with_6_distractor.csv')
if num_entities == 6:
    #inputlist_sent1,inputlist_sent2,tgtlist,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4,\
#attractorlist_5 = prepareData(dir+'job_training_introducing_6_entities.csv')
    inputlist_sent1,inputlist_sent2,tgtlist,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4,\
attractorlist_5 = prepareData(dir+'job_training_introducing_single_entity_with_5_distractor.csv')
if num_entities == 5:
    #inputlist_sent1,inputlist_sent2,tgtlist,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4,\
#= prepareData(dir+'job_training_introducing_5_entities.csv')
    inputlist_sent1,inputlist_sent2,tgtlist,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4,\
= prepareData(dir+'job_training_introducing_single_entity_with_4_distractor.csv')
if num_entities == 4:
    #inputlist_sent1,inputlist_sent2,tgtlist,attractorlist_1,attractorlist_2,attractorlist_3 = prepareData(dir+'job_training_introducing_4_entities.csv')
    inputlist_sent1,inputlist_sent2,tgtlist,attractorlist_1,attractorlist_2,attractorlist_3 = prepareData(dir+'job_training_introducing_single_entity_with_3_distractor.csv')

if num_entities == 3:
    #inputlist_sent1,inputlist_sent2,tgtlist,attractorlist_1,attractorlist_2 = prepareData(dir+'job_training_introducing_3_entities.csv')
    inputlist_sent1,inputlist_sent2,tgtlist,attractorlist_1,attractorlist_2 = prepareData(dir+'job_training_introducing_single_entity_with_2_distractor.csv')
if num_entities == 2:
    #inputlist_sent1,inputlist_sent2,tgtlist,attractorlist_1 = prepareData(dir+'job_training_introducing_2_entities.csv')
    inputlist_sent1,inputlist_sent2,tgtlist,attractorlist_1 = prepareData(dir+'job_training_introducing_single_entity_with_1_distractor.csv')

if num_entities == 7:
    test_fk_acc(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4,attractorlist_5,\
    attractorlist_6,tgtlist,bert_base,tokenizer_base,k=10)
if num_entities == 6:
    test_fk_acc(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4,attractorlist_5,\
        tgtlist,bert_base,tokenizer_base,k=10)
if num_entities == 5:
    test_fk_acc(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,attractorlist_3,attractorlist_4,\
        tgtlist,bert_base,tokenizer_base,k=10)
if num_entities == 4:
    test_fk_acc(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,attractorlist_3,\
        tgtlist,bert_base,tokenizer_base,k=10)
if num_entities == 3:
    test_fk_acc(inputlist_sent1,inputlist_sent2,attractorlist_1,attractorlist_2,\
        tgtlist,bert_base,tokenizer_base,k=10)
if num_entities == 2:
    test_fk_acc(inputlist_sent1,inputlist_sent2,attractorlist_1,\
        tgtlist,bert_base,tokenizer_base,k=10)
