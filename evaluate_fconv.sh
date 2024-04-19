#!/usr/bin/env bash
SCRIPTS=mosesdecoder/scripts
GPU=0

SRC=en
TRG=as
LANG=en-as
TEXT=wmt23.tokenized.en-as
MODEL=fconv



#(7) Decoding NMT Model
DATA_BIN=data-bin/$MODEL_${SRC}_${TRG}
CPKT=checkpoint/$MODEL_${SRC}_${TRG}
RESULT=result/$MODEL_${SRC}_${TRG}
mkdir -p $RESULT

CUDA_VISIBLE_DEVICES=$GPU fairseq-interactive $DATA_BIN \
    --path $CPKT/checkpoint_best.pt \
    --buffer-size 2000 --batch-size 128 \
    --beam 5 --remove-bpe \
    --input $TEXT/test.${SRC} | tee $RESULT/wmt23.test.${SRC}-${TRG}.${TRG}.sys

#(8) Post-processing	
# Extract translations from output file 
grep ^H $RESULT/wmt23.test.${SRC}-${TRG}.${TRG}.sys | cut -f3- > $RESULT/hypo.tok.sys

# Remove tokenization and truecasing
cat $RESULT/hypo.tok.sys | $SCRIPTS/recaser/detruecase.perl | $SCRIPTS/tokenizer/detokenizer.perl -l ${TRG} > $RESULT/hypo.sys

#(9) Automatic Evaluation
# Score the test set with sacrebleu
sacrebleu --tok=13a $RESULT/hypo.sys < en-as-test-$TRG.txt > $RESULT/sacrebleu.sys
