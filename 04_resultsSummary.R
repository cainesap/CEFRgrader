# libraries
if (!require("pacman")) install.packages("pacman", repos="https://cloud.r-project.org")
library(pacman)
pacman::p_load(tidyverse, pracma)


## arg vars ##
args <- commandArgs(trailingOnly=T)
prefix <- args[1]  # directory path to find experiment logs
## end ##


# get all expt logs in directory
files <- list.files(path=prefix, pattern='cefr-predictions_', full.names=T)


# read in each file and print expt stats
for (logfile in files) {
  # read and get expt info
  log <- as_tibble(read.csv(logfile, as.is=T))
  tstamp <- gsub('.*_([0-9]{4}.*)\\.csv', '\\1', logfile)
  date <- unlist(strsplit(tstamp, '_'))[1]
  time <- substring(gsub('[a-z]', ':', unlist(strsplit(tstamp, '_'))[2]), 1, 5)
  clfs <- unlist(strsplit(gsub('.*_[A-Za-z]{2,11}_([a-z-]*)_.*', '\\1', logfile), '-'))
  trainings <- unique(log$train)
  testings <- unique(log$test)
  pluslangs <- unique(log$pluslang)
  udv <- unique(log$UDversion)
  runs <- max(log$run)
  feats <- unique(log$features)
  n.levels <- length(unique(log$cefr))
  levellist <- paste(sort(unique(log$cefr)), collapse=', ')
  
  for (tr in trainings) {
    for (te in testings) {
      for (pl in pluslangs) {
        print('==========')
        print(paste('Experiment started on', date, 'at', time, ', training on', tr, 'testing on', te, 'pluslang:', pl, 'parsed with UD', udv, 
          'and', runs, 'runs'))
        print(paste('With', n.levels, 'CEFR levels:', levellist))
        
        # results per feature
        for (ft in feats) {
          feat.df <- log %>% filter(features==ft, train==tr, test==te, pluslang==pl)  # subset
          true <- feat.df$cefr  # get true CEFR levels
          for (clf in clfs) {
            clf.df <- as_tibble(feat.df[,which(colnames(feat.df)==clf)])  # get classifier predictions column
            clf.df <- clf.df %>% add_column(true)  # add true levels
            colnames(clf.df)[1] <- 'pred'  # rename
            clf.df$pred <- as.factor(clf.df$pred)
            clf.df$true <- as.factor(clf.df$true)
            # f1 per CEFR level for weighted-F1
            f1s <- c()
            weights <- c()
            for (cefr in levels(clf.df$true)) {
              n.true <- clf.df %>% filter(true==cefr) %>% nrow()  # n.true instances of this CEFR level
              weights <- append(weights, n.true)  # add as weight
              cefr.pred <- clf.df %>% filter(pred==cefr)  # subset predicted at this CEFR level
              tp <- cefr.pred %>% filter(true==cefr) %>% nrow()  # true positives
              prec <- tp / nrow(cefr.pred)  # precision
              rec <- tp / n.true  # recall
              f1 <- 2*( (prec*rec) / (prec+rec) )  # F1 = harmonic mean of prec + rec
              f1s <- append(f1s, f1)
            }
            f1s[is.na(f1s)] <- 0  # replace any NaNs with 0
            wf1 <- round(pracma::dot(f1s, weights) / nrow(clf.df), 4)  # weighted F1 calc
            # go numeric for RMSE and %within1
            clf.df$pred <- as.numeric(clf.df$pred)
            clf.df$true <- as.numeric(clf.df$true)
            sqerr <- (clf.df$true-clf.df$pred)^2
            nrows <- nrow(clf.df)
            within1 <- round((nrows-sum(sqerr>1)) / nrows, 4)
            rmse <- round(sqrt(mean(sqerr)), 4)
            # info print
            print(paste(tr, 'train', te, 'test: weighted-F1 with', ft, 'feature(s)', toupper(clf), 'classifier =', wf1,
              ', RMSE =', rmse, ', %within1 =', within1))
          }
        }
      }
    }
  }
}
