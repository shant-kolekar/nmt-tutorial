#!/usr/bin/env bash
SCRIPTS=mosesdecoder/scripts
GPU=2

SRC=en
TRG=as
TEXT=wmt23.tokenized.en-as

DATA_BIN=data-bin/xfmr_${SRC}_${TRG}
mkdir -p $DATA_BIN

#(5) Word to Integer Sequence
fairseq-preprocess --source-lang ${SRC} --target-lang ${TRG} \
    --trainpref $TEXT/train --validpref $TEXT/valid \
    --destdir $DATA_BIN \
    --workers 20
	
CPKT=checkpoint/xfmr_${SRC}_${TRG}
LOG=log/xfmr_${SRC}_${TRG}
mkdir -p  $CPKT $LOG


#(6) Train NMT Model (Transformer)
CUDA_VISIBLE_DEVICES=$GPU fairseq-train $DATA_BIN \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-epoch 50 --patience 5 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir $CPKT --no-epoch-checkpoints | tee $LOG/train_transformer.out