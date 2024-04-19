#!/usr/bin/env bash
SCRIPTS=mosesdecoder/scripts
GPU=0

SRC=en
TRG=as
LANG=en-as
TEXT=wmt23.tokenized.en-as

DATA_BIN=data-bin/fconv_${SRC}_${TRG}
mkdir -p $DATA_BIN

#(5) Word to Integer Sequence
fairseq-preprocess --source-lang ${SRC} --target-lang ${TRG} \
    --trainpref $TEXT/train --validpref $TEXT/valid \
    --destdir $DATA_BIN \
    --workers 20
	
CPKT=checkpoint/fconv_${SRC}_${TRG}
LOG=log/fconv_${SRC}_${TRG}
mkdir -p  $CPKT $LOG


#(6) Train NMT Model (Convolutional Seq2Seq)
CUDA_VISIBLE_DEVICES=$GPU fairseq-train $DATA_BIN \
  --lr 0.25 --clip-norm 0.1 --dropout 0.2 \
  --max-tokens 4000 \
  --arch fconv_iwslt_de_en \
  --criterion label_smoothed_cross_entropy \
  --optimizer nag --label-smoothing 0.1 \
  --lr-scheduler fixed --force-anneal 50 \
  --max-epoch 50 --patience 5 \
  --save-dir $CPKT --no-epoch-checkpoints | tee $LOG/train_fconv.out