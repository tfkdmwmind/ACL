#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 09:13:12 2018

@author: xcq
"""
import json
from coco.pycocotools.coco import COCO
from coco.pycocoevalcap.eval import COCOEvalCap
import torch
from utils import coco_eval,generate_attention_map
import pickle

from model import get_model
from build_vocab import Vocabulary
from config import train_config
import os

caption_val_path='./data/annotations/karpathy_split_val.json'
result_path='./results'
def eval_model_list(caption_val_path,model_path,args):
    result={}
    for model in os.listdir(model_path):
        cider=eval_model(caption_val_path,os.path.join(model_path,model),args)
        result[model]=cider
    return result
def eval_model(caption_val_path,model_path,args):
    with open( args.vocab_path, 'rb') as f:
        vocab = pickle.load( f )
        print len(vocab)

    adaptive=get_model(args,len(vocab))

    adaptive.load_state_dict( torch.load( model_path ) )
    if torch.cuda.is_available():
        adaptive.cuda()
    epoch = int( model_path.split('/')[-1].split('-')[1].split('.')[0] )
#    cider = coco_eval( adaptive, args, epoch )
    cider=generate_attention_map(adaptive,args,epoch)
    return cider
if __name__=='__main__':
    args=train_config()
    args.model='adaptive'
    result=eval_model(caption_val_path,'models/final-result/adaptive/adaptive-20.pkl',args)