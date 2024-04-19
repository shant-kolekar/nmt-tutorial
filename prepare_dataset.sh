# Set paths for scripts
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
TCROOT=$SCRIPTS/recaser/
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=16000

# Prepare directories
SRC=en
TRG=as
LANG=en-as
PREP=wmt23.tokenized.en-as
TMP=$PREP/tmp
ORIG=orig


mkdir -p $ORIG $TMP $PREP

## Tokenizing Train Data
echo "Tokenizing train data..."
cat en-as-train-$SRC.txt | $TOKENIZER -threads 8 -a -l $SRC > $TMP/train.tags.$LANG.tok.$SRC
echo ""

cat en-as-train-$TRG.txt | $TOKENIZER -threads 8 -a -l $TRG > $TMP/train.tags.$LANG.tok.$TRG
echo ""

## Tokenizing Valid data
echo "Tokenizing valid data..."
cat en-as-valid-$SRC.txt | $TOKENIZER -threads 8 -a -l $SRC > $TMP/valid.tags.$LANG.tok.$SRC
echo ""

cat en-as-valid-$TRG.txt | $TOKENIZER -threads 8 -a -l $TRG > $TMP/valid.tags.$LANG.tok.$TRG
echo ""

# Clean Train Data
echo "Cleaning train data..."
perl $CLEAN -ratio 1.5 $TMP/train.tags.$LANG.tok $SRC $TRG $TMP/train.tok.clean 1 175

# Clean Valid Data
echo "Cleaning valid data..."
perl $CLEAN -ratio 1.5 $TMP/valid.tags.$LANG.tok $SRC $TRG $TMP/valid.tok.clean 1 175

# Truecase Data
perl $TCROOT/train-truecaser.perl -corpus $TMP/train.tok.clean.$SRC -model $PREP/truecase-model.$SRC
perl $TCROOT/train-truecaser.perl -corpus $TMP/train.tok.clean.$TRG -model $PREP/truecase-model.$TRG


echo "Truecasing train data"
for L in $SRC $TRG
 do
	$TCROOT/truecase.perl -model $PREP/truecase-model.$L < $TMP/train.tok.clean.$L > $TMP/train.tc.$L
done

echo "Truecasing valid data"
for L in $SRC $TRG 
 do
	$TCROOT/truecase.perl -model $PREP/truecase-model.$L < $TMP/valid.tok.clean.$L > $TMP/valid.tc.$L
done

# Byte Pair Encoding
TRAIN=$TMP/train.$LANG
BPE_CODE=$PREP/code
rm -f $TRAIN

echo "learn_bpe.py on ${TRAIN}.${SRC}"
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TMP/train.tc.$SRC > $BPE_CODE.$SRC.bpe

echo "learn_bpe.py on ${TRAIN}.${TRG}"
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TMP/train.tc.$TRG > $BPE_CODE.$TRG.bpe


for L in $SRC $TRG
 do
    for F in train valid
     do
        echo "apply_bpe.py on ${F}"
        python $BPEROOT/apply_bpe.py -c $BPE_CODE.$L.bpe < $TMP/$F.tc.$L > $PREP/$F.$L
    done
done


echo "Tokenize test data"
cat en-as-test-$SRC.txt | $TOKENIZER -threads 8 -a -l $SRC > $TMP/test.tok.$SRC

echo "Truecase test.$SRC"
$TCROOT/truecase.perl -model $PREP/truecase-model.$SRC < $TMP/test.tok.$SRC > $TMP/test.tc.$SRC

echo "Apply bpe on test.$SRC"
python $BPEROOT/apply_bpe.py -c $BPE_CODE.$SRC.bpe < $TMP/test.tc.$SRC > $PREP/test.$SRC

# echo "Removing temporary directory"
# rm -rf $TMP