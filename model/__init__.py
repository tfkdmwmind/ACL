#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 17:53:42 2018

@author: xcq
"""

from adaptive import Adaptive
from myattentive import Myattentive
from en2de import Encoder2Decoder
from attentive import Attentive

def get_model(args,vocab_size):
    if args.model=='adaptive':
        adaptive=Adaptive(args.embed_size,vocab_size,args.hidden_size)
    elif args.model=='adaptive-mean':
        adaptive=Adaptive(args.embed_size,vocab_size,args.hidden_size)
    elif args.model=='en2de':
        adaptive=Encoder2Decoder(512,vocab_size,args.hidden_size)
    elif args.model=='en2de-mean':
        adaptive=Encoder2Decoder(512,vocab_size,args.hidden_size)
    elif args.model=='attentive':
        adaptive=Attentive(args.embed_size,vocab_size,args.hidden_size)
    elif args.model=='attentive-mean':
        adaptive=Attentive(args.embed_size,vocab_size,args.hidden_size)
    return adaptive
