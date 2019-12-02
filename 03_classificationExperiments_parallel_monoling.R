## REPROLANG replication of Vajjala & Rama 2018, Experiments with Universal CEFR Classification, BEA.
## ACL Anthology: https://www.aclweb.org/anthology/W18-0515
## V&R's code repo: https://github.com/ nishkalavallabhi/UniversalCEFRScoring


## PRELIMS
print('Loading libraries...')

# libraries: for parallel compute
if (!require("pacman")) install.packages("pacman", repos="https://cloud.r-project.org")
library(pacman)
pacman::p_load(doSNOW, foreach)


## arg vars ##
args <- commandArgs(trailingOnly=T)
prefix <- args[1]  # filepath for pre-processed MERLIN features
## end ##



## MONOLINGUAL experiments (Table 2 in V&R paper)
print('==========')
print('Running monolingual experiments...')

# experiments on which languages?
languages <- c('CZ', 'DE', 'IT')

# grab 1 cluster per language
nCore <- length(languages)
print(paste('Starting cluster of', nCore, 'cores...'))
clust <- makeCluster(nCore, outfile='')
registerDoSNOW(clust)


# parallel foreach function
langs <- foreach (lg=languages, .combine='c', .packages=c('tidyverse')) %dopar% {
  
  # functions to run experiments
  print('Sourcing experiment functions...')
  source('R/caretClassify.R')
  source('R/kerasClassify.R')
  
  
  ## DATA
  # load pre-processed features, texts, testfolds
  print('Loading data...')
  # which UDPipe model?
  udv <- '2.0'  # used by V&R
#  udv <- '2.4'  # latest at time of writing
  merlin <- read_tsv(paste0(prefix, 'merlin_features_udpipe', udv, '.tsv'))
  merlin$cefr <- as.factor(merlin$cefr)
  texts <- read_tsv(paste0(prefix, 'merlin_texts.tsv'))
  texts$cefr <- as.factor(texts$cefr)
  fold.df <- read_tsv(paste0(prefix, 'merlin_folds.tsv'))
  fold.df$cefr <- as.factor(fold.df$cefr)
  
  # run log.reg, RFs, SVM as in V&R
  classifiers <- c('lr', 'rf', 'svm')
  # run XGBoost, Keras MLP (time / performance costs respectively)
#  classifiers <- c('xgb', 'ker')
  
  # log-file
  exptstart <- format(Sys.time(), format="%Y-%m-%d_%Hh%Mm%Ss")
  fileout <- paste0(prefix, '/logs/experiment-summary_', lg, '_', paste(classifiers, collapse='-'), '_', exptstart, '.csv')
  foldfile <- paste0(prefix, '/logs/cefr-predictions_', lg, '_', paste(classifiers, collapse='-'), '_', exptstart, '.csv')
  
  # subset data, make wide, select features, filter by doc.freq
  print(paste('Working with language', lg, ': filtering, reshaping, compiling doc.freq counts...'))
  lang.df <- subset(merlin, lang==lg)
  text.df <- subset(texts, lang==lg)
  folds <- subset(fold.df, lang==lg)
  
  # i: baseline doc.length only
  append <- 0  # first expt
  feats <- 'doc.length'
  mdf <- 0
  pluslang <- FALSE  # involving language as a feature/prediction.task
  trainEval(lang.df, classifiers, lg, lg, feats, mdf, fileout, foldfile, append, folds, pluslang, udv)
  
  # ii: word ngrams
  append <- 1  # append from now on
  feats <- 'word'
  mdf <- 1  # minimum document frequency for feature inclusion (tuned in function to <=1000 feats)
  trainEval(lang.df, classifiers, lg, lg, feats, mdf, fileout, foldfile, append, folds, pluslang, udv)
  
  # iii: pos ngrams
  feats <- 'pos'
  trainEval(lang.df, classifiers, lg, lg, feats, mdf, fileout, foldfile, append, folds, pluslang, udv)
  
  # iv: dependency ngrams
  feats <- 'dep.triples'
  trainEval(lang.df, classifiers, lg, lg, feats, mdf, fileout, foldfile, append, folds, pluslang, udv)
  
  # v: domain features
  feats <- 'domain'
  mdf <- 0
  trainEval(lang.df, classifiers, lg, lg, feats, mdf, fileout, foldfile, append, folds, pluslang, udv)
  
  # vi: word ngrams + domain features
  feats <- 'domain+word'
  mdf <- 1
  trainEval(lang.df, classifiers, lg, lg, feats, mdf, fileout, foldfile, append, folds, pluslang, udv)
  
  # vii: pos ngrams + domain features
  feats <- 'domain+pos'
  trainEval(lang.df, classifiers, lg, lg, feats, mdf, fileout, foldfile, append, folds, pluslang, udv)
  
  # viii: dependency ngrams + domain features
  feats <- 'domain+dep.triples'
  trainEval(lang.df, classifiers, lg, lg, feats, mdf, fileout, foldfile, append, folds, pluslang, udv)
  
  # ix: word embeddings with keras (runs 10 times)
  append <- 0  # new logfile
  mwf <- 15  # from V&R 'code/monolingual_cv.p' word freq > 15
  foldfile <- paste0(prefix, '/logs/cefr-predictions_', lg, '_keras_', exptstart, '.csv')
  kerasEval(text.df, lg, lg, mwf, fileout, foldfile, append, folds, prefix, pluslang, udv)
  
  # return
  lg
}

print('==========')
print(paste('Finished, see logs in:', prefix))
