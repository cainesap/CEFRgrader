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



print('Sourcing experiment functions...')
source('R/caretClassify.R')


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


## CROSS-LINGUAL experiments (Table 4 in V&R paper)
print('==========')
print('Running cross-lingual experiments...')

# new logfiles
exptstart <- format(Sys.time(), format="%Y-%m-%d_%Hh%Mm%Ss")
fileout <- paste0(prefix, 'logs/experiment-summary_crossling_', paste(classifiers, collapse='-'), '_', exptstart, '.csv')
foldfile <- paste0(prefix, 'logs/cefr-predictions_crossling_', paste(classifiers, collapse='-'), '_', exptstart, '.csv')

# expt set up: train on DE, test on CZ and IT
append <- 0  # single logfile for both test langs
pluslang <- FALSE
train <- 'crossling'
langs <- c('CZ', 'IT')

for (test in langs) {
  
  # i: baseline doc.length only
  feats <- 'doc.length'
  mdf <- 0
  trainEval(merlin, classifiers, train, test, feats, mdf, fileout, foldfile, append, folds, pluslang, udv)
  
  # ii: word ngrams (not run by V&R)
  # iii: pos ngrams
  append <- 1  # append from now on
  feats <- 'pos'
  mdf <- 1  # minimum document frequency for feature inclusion (tuned in function to <=1000 feats)
  trainEval(merlin, classifiers, train, test, feats, mdf, fileout, foldfile, append, folds, pluslang, udv)
  
  # iv: dependency ngrams
  feats <- 'dep.triples'
  trainEval(merlin, classifiers, train, test, feats, mdf, fileout, foldfile, append, folds, pluslang, udv)
  
  # v: domain features
  feats <- 'domain'
  mdf <- 0
  trainEval(merlin, classifiers, train, test, feats, mdf, fileout, foldfile, append, folds, pluslang, udv)
}
