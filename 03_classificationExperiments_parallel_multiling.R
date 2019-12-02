## REPROLANG replication of Vajjala & Rama 2018, Experiments with Universal CEFR Classification, BEA.
## ACL Anthology: https://www.aclweb.org/anthology/W18-0515
## V&R's code repo: https://github.com/ nishkalavallabhi/UniversalCEFRScoring


## PRELIMS
print('Loading libraries...')

# libraries: the tidyverse
if (!require("pacman")) install.packages("pacman")
library(pacman)
pacman::p_load(tidyverse)


## arg vars ##
args <- commandArgs(trailingOnly=T)
prefix <- args[1]  # filepath for pre-processed MERLIN features; will save experiment logs here too
## end ##



## MULTILINGUAL experiments (Table 3 in V&R paper)
print('==========')
print('Running multilingual experiments...')

print('Sourcing experiment functions...')
source('R/caretClassify.R')
source('R/kerasClassify.R')

# which UDPipe model?
udv <- '2.0'  # used by V&R
#udv <- '2.4'  # latest at time of writing

# read data
merlin <- read_tsv(paste0(prefix, 'merlin_features_udpipe', udv, '.tsv'))
merlin$cefr <- as.factor(merlin$cefr)
texts <- read_tsv(paste0(prefix, 'merlin_texts.tsv'))
texts$cefr <- as.factor(texts$cefr)
folds <- read_tsv(paste0(prefix, 'merlin_folds.tsv'))
folds$cefr <- as.factor(folds$cefr)

# run log.reg, RFs, SVM as in V&R
classifiers <- c('lr', 'rf', 'svm')

# expt set up: repeat with/without language as feature/objective
lang <- 'multiling'
pluslangs <- c(TRUE, FALSE)

for (pluslang in pluslangs) {
  
  # new logfiles
  exptstart <- format(Sys.time(), format="%Y-%m-%d_%Hh%Mm%Ss")
  fileout <- paste0(prefix, 'logs/experiment-summary_multiling_', paste(classifiers, collapse='-'), '_', exptstart, '.csv')
  foldfile <- paste0(prefix, 'logs/cefr-predictions_multiling_', paste(classifiers, collapse='-'), '_', exptstart, '.csv')
  
  # i: baseline doc.length only
  append <- 0  # separate logfiles because Keras
  feats <- 'doc.length'
  mdf <- 0
  trainEval(merlin, classifiers, lang, lang, feats, mdf, fileout, foldfile, append, folds, pluslang, udv)
  
  # ii: word ngrams
  append <- 1  # append from now on
  feats <- 'word'
  mdf <- 1  # minimum document frequency for feature inclusion (tuned in function to <=1000 feats)
  trainEval(merlin, classifiers, lang, lang, feats, mdf, fileout, foldfile, append, folds, pluslang, udv)
  
  # iii: pos ngrams
  feats <- 'pos'
  trainEval(merlin, classifiers, lang, lang, feats, mdf, fileout, foldfile, append, folds, pluslang, udv)
  
  # iv: dependency ngrams
  feats <- 'dep.triples'
  trainEval(merlin, classifiers, lang, lang, feats, mdf, fileout, foldfile, append, folds, pluslang, udv)
  
  # v: domain features
  feats <- 'domain'
  mdf <- 0
  trainEval(merlin, classifiers, lang, lang, feats, mdf, fileout, foldfile, append, folds, pluslang, udv)
  
  # vi: word embeddings with keras (runs 10 times)
  append <- 0
  mwf <- 15  # from V&R 'code/monolingual_cv.p' word freq > 15
  foldfile <- paste0(prefix, 'logs/cefr-predictions_multiling_keras_', exptstart, '.csv')
  kerasEval(texts, lang, lang, mwf, fileout, foldfile, append, folds, prefix, pluslang, udv)
}

