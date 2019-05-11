#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:21:16 2018

@author: xcq
"""

class train_config(object):
    def __init__(self):
        self.f='self'
        self.crop_size=224
        self.vocab_path='./data/vocab.pkl'
        self.image_dir='./data/resized'
        self.caption_path='./data/annotations/karpathy_split_train.json'
        self.caption_val_path='./data/annotations/karpathy_split_val.json'
        self.log_step=10
        self.seed=123
        #Hyper parameter setup
        self.fine_tune_start_layer=5
        self.cnn_epoch=20
        self.alpha=0.8
        self.beta=0.999
        self.learning_rate=4e-4
        self.learning_rate_cnn=1e-4
        #LSTM hyper parameters
        self.embed_size=256
        self.hidden_size=512
        self.num_epochs=50
        self.batch_size=42
        self.eval_size=28
        self.num_workers=4
        self.clip=0.1
        self.lr_decay=20
        self.learning_rate_decay_every=50
        #
        self.model='adaptive'
        self.model_path='./models/final-result/adaptive-mean'
        self.pretrained='./models/final-result/adaptive-mean/adaptive-20.pkl'
        
#        self.model='attentive-mean'
#        self.model_path='./models/final-result/attentive-mean'
#        self.pretrained='./models/final-result/attentive-mean/adaptive-20.pkl'
        
#        self.model='attentive-mean'
#        self.model_path='./models/final-result/attentive-mean'
#        self.pretrained='./models/final-result/attentive-mean/adaptive-20.pkl'
        
#        self.model='en2de-mean'
#        self.model_path='./models/final-result/en2de'
#        self.pretrained='./models/final-result/en2de/adaptive-20.pkl'

#        self.model='en2de'
#        self.model_path='./models/final-result/en2de-mean'
#       	self.pretrained='./models/final-result/en2de-mean/adaptive-20.pkl'
#        self.pretrained=None
