import math
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from utils import coco_eval, to_var
from data_loader import get_loader 
from model import *
from build_vocab import Vocabulary
from torch.autograd import Variable 
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from loss import CrossEntropyLoss_mul,CrossEntropyLoss_mean
from config import train_config
    
def main(args):
    
    # To reproduce training results
    torch.manual_seed( args.seed )
    if torch.cuda.is_available():
        torch.cuda.manual_seed( args.seed )
        
    # Create model directory
    if not os.path.exists( args.model_path ):
        os.makedirs(args.model_path)
    
    # Image Preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([ 
        transforms.RandomCrop( args.crop_size ),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(( 0.485, 0.456, 0.406 ), 
                             ( 0.229, 0.224, 0.225 ))])
    
    # Load vocabulary wrapper.
    with open( args.vocab_path, 'rb') as f:
        vocab = pickle.load( f )
    
    # Build training data loader
    data_loader = get_loader( args.image_dir, args.caption_path, vocab, 
                              transform, args.batch_size,
                              shuffle=True, num_workers=args.num_workers ) 

    # Load pretrained model or build from scratch
    adaptive=get_model(args,len(vocab))
    
    if args.pretrained:
        
        adaptive.load_state_dict( torch.load( args.pretrained ) )
        # Get starting epoch #, note that model is named as '...your path to model/algoname-epoch#.pkl'
        # A little messy here.
        start_epoch = int( args.pretrained.split('/')[-1].split('-')[1].split('.')[0] ) + 1
        
    else:
        start_epoch = 1
    
    # Constructing CNN parameters for optimization, only fine-tuning higher layers
    cnn_params=adaptive.cnn_params(args.fine_tune_start_layer)
    cnn_optimizer = torch.optim.Adam( cnn_params, lr=args.learning_rate_cnn, 
                                      betas=( args.alpha, args.beta ) )
    
    #rnn params
    params=adaptive.rnn_params()

    # Will decay later    
    learning_rate = args.learning_rate
    
    # Language Modeling Loss
    if 'mean' in args.model:
        LMcriterion=CrossEntropyLoss_mean()
    else:
        LMcriterion=nn.CrossEntropyLoss()
    test_loss_fun=nn.CrossEntropyLoss()
    # Change to GPU mode if available
    if torch.cuda.is_available():
        adaptive.cuda()
        LMcriterion.cuda()
        test_loss_fun.cuda()
        
    # Train the Models
    total_step = len( data_loader )
    
    cider_scores = []
    best_cider = 0.0
    best_epoch = 0
    
    # Start Training
    for epoch in range( start_epoch, args.num_epochs + 1 ):

        # Start Learning Rate Decay
        if epoch > args.lr_decay:
                
            frac = float( epoch - args.lr_decay ) / args.learning_rate_decay_every
            decay_factor = math.pow( 0.5, frac )

            # Decay the learning rate
            learning_rate = args.learning_rate * decay_factor
        
        print 'Learning Rate for Epoch %d: %.6f'%( epoch, learning_rate )

        optimizer = torch.optim.Adam( params, lr=learning_rate, betas=( args.alpha, args.beta ) )
        # Language Modeling Training
        total_collect=0

        print '------------------Training for Epoch %d----------------'%( epoch )
        for i, (images_, captions_, lengths_, _, _ ) in enumerate( data_loader ):
            # Set mini-batch dataset

            images = to_var( images_ )
            captions = to_var( captions_ )
            lengths = [ cap_len - 1  for cap_len in lengths_ ]
            targets = pack_padded_sequence( captions[:,1:], lengths, batch_first=True )[0]

            # Forward, Backward and Optimize
            adaptive.train()
            adaptive.zero_grad()
            packed_scores = adaptive( images, captions, lengths )
            
            #compute correct num of words
            pad=pad_packed_sequence(packed_scores,batch_first=True)
            num_correct=0
            for ids in range(len(lengths)):
                cap_len=lengths[ids]
                pred=pad[0][ids][:cap_len].max(1)[1]
                ground=captions[ids][1:cap_len+1]
                correct=np.sum(pred.data.cpu().numpy()==ground.data.cpu().numpy())
                num_correct=num_correct+correct
            total_collect+=num_correct
            correct_prop=float(num_correct)/sum(lengths)
            # Compute loss and backprop
            if 'mean' in args.model:
                loss=LMcriterion(packed_scores,targets,lengths)
            else:
                loss = LMcriterion( packed_scores[0], targets )
            loss.backward()
            
            test_loss=test_loss_fun(packed_scores[0],targets)
            adaptive.clip_params(args.clip)
            
            optimizer.step()
            
            # Start CNN fine-tuning
            if epoch > args.cnn_epoch:

                cnn_optimizer.step()
            
            # Print log info
            if i % args.log_step == 0:
                print 'Epoch [%d/%d], Step [%d/%d], CrossEntropy Loss: %.4f, Perplexity: %5.4f\ncorrect_prop:%5.4f,test_loss:%.4f'%( epoch, 
                                                                                                 args.num_epochs, 
                                                                                                 i, total_step, 
                                                                                                 loss.data[0],
                                                                                                 np.exp( loss.data[0] ),
                                                                                                 correct_prop,
                                                                                                test_loss)

        # Save the Adaptive Attention model after each epoch
        torch.save( adaptive.state_dict(), 
                    os.path.join( args.model_path, 
                    'adaptive-%d.pkl'%( epoch ) ) )          

        # Evaluation on validation set        
        eval_result = coco_eval( adaptive, args, epoch )
        eval_result['collect_num']=total_collect
        cider=eval_result['CIDEr']
        cider_scores.append( cider )

        result_path='results-score/'+args.model
        if not os.path.exists(result_path):
            re={}
            re['adaptive-%d.pkl'%epoch]=eval_result
            json.dump(re,open(result_path,'w'))
        else:
            re=json.load(open(result_path,'r'))
            re['adaptive-%d.pkl'%epoch]=eval_result
            json.dump(re,open(result_path,'w'))

        if cider > best_cider:
            best_cider = cider
            best_epoch = epoch
       
        if len( cider_scores ) > 5 and epoch>20:
            
            last_6 = cider_scores[-6:]
            last_6_max = max( last_6 )
            
            # Test if there is improvement, if not do early stopping
            if last_6_max != best_cider:
                
                print 'No improvement with CIDEr in the last 6 epochs...Early stopping triggered.'
                print 'Model of best epoch #: %d with CIDEr score %.2f'%( best_epoch, best_cider )
                break
            
            
if __name__ == '__main__':
    args=train_config()

    print '------------------------training %s--------------------------'%args.model
    # Start training
    rest=main( args )
