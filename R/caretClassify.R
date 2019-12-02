## REPROLANG replication of Vajjala & Rama 2018, Experiments with Universal CEFR Classification, BEA.
## ACL Anthology: https://www.aclweb.org/anthology/W18-0515
## V&R's code repo: https://github.com/ nishkalavallabhi/UniversalCEFRScoring



## PRELIMS

# libraries: udpipe for parsing, tidytext for n-grams, magrittr for fwds pipe %>%
if (!require("pacman")) install.packages("pacman")
library(pacman)
pacman::p_load(tidyverse, caret, nnet, randomForest, LiblineaR, performanceEstimation, scales, xgboost, keras, e1071)



# function to transform long to wide data
dataTransform <- function(df, features, mindocfreq, target.lang) {
  # build feature lists
  extrafts <- c()
  # first: if feature combination, don't overwrite features obj just yet, set domain feats aside
  if (grepl('\\+', features)) {
    extrafts <- c('doc.length', 'lex.density', 'lex.variation', 'lex.diversity', 'gramm.error', 'spell.error')
  } else if (grepl('domain', features)) {  # if no + and contains 'domain'
    features <- c('doc.length', 'lex.density', 'lex.variation', 'lex.diversity', 'gramm.error', 'spell.error')
  }
  if (grepl('word', features)) {
    features <- paste0('word', 1:5)
  } else if (grepl('pos', features)) {
    features <- paste0('pos', 1:5)
  } else if (grepl('dep', features)) {
    features <- 'dep.triples'  # ensure drop '+domain' from 'dep.triples'
  }
  # select features  
  df.subs <- subset(df, type %in% features)
  # add doc frequency
  df.df <- df.subs %>% add_count(feats, name='doc.freq') %>% as.data.frame()
  # set doc.vars to df=0
  df.df$doc.freq[which(df.df$type %in%
    c('doc.length', 'lex.density', 'lex.variation', 'lex.diversity', 'gramm.error', 'spell.error'))] <- 0
  # filter by min.doc.freq if required
  if (mindocfreq>0) {
    # tune threshold to keep n.dimensions manageable
    stopping <- 0
    while (stopping==0) {
      uniq <- length(unique(subset(df.df, doc.freq>=mindocfreq)$feats))
#      if (uniq<=1000) {  # main expts
      if (uniq<=400) {  # reduced for efficiency
        stopping <- 1
      } else {  # or add one to mdf
        mindocfreq <- mindocfreq+1
      }
    }
    print(paste('Language', target.lang, 'using min.doc.frequency of', mindocfreq))
    # subset by min.doc.freq threshold (doc.feats have doc.freq of 0)
    df.subs <- subset(df.df, doc.freq==0 | doc.freq>=mindocfreq)
  }
  # make CEFR factor again, drop empty levels
  df.subs$cefr <- as.factor(df.subs$cefr)
  df.subs$cefr <- droplevels(df.subs$cefr)
  # make wide: column names from features, values from 'value'
  df.piv <- df.subs %>% pivot_wider(names_from=feats, values_fill=list(value=0), names_repair='universal')  # zero not NA for missing values
  # put single row of doc.id, lang, cefr to one side
  df.doc <- df.piv %>% select(doc.id, lang, cefr) %>% distinct(doc.id, .keep_all=T)
  df.doc$mindocfreq <- mindocfreq  # add record of min.doc.freq
  # group by document and sum n-gram counts, scale values 0..1, recombine with metadata
  df.wide<- df.piv %>% group_by(doc.id) %>% summarize_if(is.numeric, sum) %>% select(-c(doc.id)) %>% mutate_all(rescale) %>%
    cbind(df.doc) %>% as_tibble()
  # if feature combination, get class probabilites from n-gram only model, combine with domain feats
  if (length(extrafts)>0) {
    print(paste('Language', target.lang, 'calculating class probabilities for', gsub('[0-9]+', '', features[1]), 'features:'))
    df.wide$testfold <- createFolds(df.wide$cefr, k=10, list=F)
    class.probs <- data.frame()
#    for (fld in 1:2) {  # test
    for (fld in 1:10) {  # for each fold
      print(paste('Calculating class probs for', target.lang, gsub('[0-9]+', '', features[1]), 'features: fold', fld, '...'))
      tr <- subset(df.wide, testfold!=fld)
      te <- subset(df.wide, testfold==fld)
      tr <- tr %>% select(-c(doc.id, lang, mindocfreq, testfold))
      ids <- te$doc.id  # store doc.id
      te <- te %>% select(-c(doc.id, lang, cefr, mindocfreq, testfold))
      mod <- train(cefr ~ ., data=tr, method='rf', verbose=F)  # randomForest as in V&R code
      prob.df <- predict(mod, te, type='prob')
      prob.df$doc.id <- ids
      class.probs <- rbind(class.probs, prob.df)
    }
    # mean class probs for each doc
    dense.feats <- class.probs %>% group_by(doc.id) %>% summarize_all(mean)
    # now get domain features, reshape, rescale
    df.subs <- subset(df, type %in% extrafts)
    df.subs$cefr <- as.factor(df.subs$cefr)
    df.subs$cefr <- droplevels(df.subs$cefr)
    df.piv <- df.subs %>% pivot_wider(names_from=feats, values_fill=list(value=0), names_repair='universal')  # zero not NA for missing values
    df.doc <- df.piv %>% select(doc.id, lang, cefr) %>% distinct(doc.id, .keep_all=T)
    df.doc$mindocfreq <- mindocfreq  # add record of min.doc.freq
    # group into a single row per document, scale 0..1 and append to metadata
    df.wide <- df.piv %>% group_by(doc.id) %>% summarize_if(is.numeric, max) %>% select(-c(doc.id)) %>%
      mutate_all(rescale) %>% cbind(df.doc) %>% as_tibble()
    # combine domain and dense n-gram features
    df.wide <- merge(df.wide, dense.feats, by='doc.id')
  }
  df.wide  # return
}


# function to train and test, store results
trainEval <- function(in.df, clfs, train, test, features, mindf, fout, foldout, app, folds, pluslang, udv) {
  # data transform: subset, make wide, select features, filter by doc.freq
  print('========')
  # drop unused CEFR levels
  in.df$cefr <- droplevels(in.df$cefr)
  lang <- train
  print(paste('Language', lang, 'preparing data...'))
  if (grepl('crossling', train)) {
    # CROSSLING EXPTS: train on DE, test on CZ and IT (no folds)
    train <- 'DE'
    tr.df <- in.df %>% filter(lang==train)
    te.df <- in.df %>% filter(lang==test)
    tt.df <- rbind(tr.df, te.df)
    tt <- dataTransform(tt.df, features, mindf, lang)
    tt$testfold <- 0
    tt$testfold[which(tt$lang==test)] <- 1  # only test on test set
  } else {
    # MONOLINGUAL EXPTS: train and test the same
    # ADVERSARIAL EXPTS: train and test folds, Spanish is always test
    # MULTILING EXPTS: train and test folds for all langs
    tt <- dataTransform(in.df, features, mindf, lang)
    # get test folds from pre-processed table
    tt$testfold <- 0
    for (i in 1:nrow(tt)) {
      did <- tt$doc.id[i]
      tt$testfold[i] <- folds$testfold[which(folds$doc.id==did)]
    }
  }
  # get record of updated min.doc.freq
  mindf <- tt$mindocfreq[1]
  # train and evaluate
  eval.df <- data.frame()
  for (i in 1:1) {  # test: run thru once
#  for (i in 1:10) {  # average 10 runs
    print(paste('Language', lang, ': start time for this run @', as.character(Sys.time())))
#    for (fld in 1:2) {  # test
    for (fld in 1:max(tt$testfold)) {  # for each fold
      # get train and test
      tr <- subset(tt, testfold!=fld)
      te <- subset(tt, testfold==fld | testfold==-1)  # add -1 adversarial Spanish texts to test
      # grab test truth values and start folds output table
      truth <- te$cefr
      fold.df <- te %>% select(doc.id, testfold, lang, cefr)
      fold.df$min.freq <- mindf
      fold.df$min.freq.type <- 'doc.freq'
      fold.df$run <- i
      fold.df$train <- train
      fold.df$test <- test
      fold.df$features <- features
      fold.df$UDversion <- udv
      # drop unnecessary columns with dplyr select(), depending on lang+/-
      if (pluslang) {
        tr$lang <- as.factor(tr$lang)  # keep language as factor variable
        tr <- tr %>% select(-c(doc.id, mindocfreq, testfold))
        te <- te %>% select(-c(doc.id, cefr, mindocfreq, testfold))
      } else {
        tr <- tr %>% select(-c(doc.id, lang, mindocfreq, testfold))
        te <- te %>% select(-c(doc.id, lang, cefr, mindocfreq, testfold))
      }
      vsize <- ncol(tr)-1  # n.features
      fold.df$vocab_size <- vsize
      fold.df$pluslang <- pluslang
      # for each classifier (caret ML: see https://topepo.github.io/caret/train-models-by-tag.html)
      for (clf in clfs) {
        start <- as.integer(Sys.time())  # epoch seconds
        print('========')
        print(paste('Training run', i, 'fold', fld, ':', clf, 'classifier and', features,
          'features on', train, 'language data...'))
        if (clf=='lr') {
          messages <- capture.output(mod <- train(cefr ~ ., data=tr, method='multinom', verbose=F, maxit=1000, MaxNWts=20000))  # nnet
        } else if (clf=='rf') {
          mod <- train(cefr ~ ., data=tr, method='rf', verbose=F)  # randomForest
        } else if (clf=='svm') {
          mod <- train(cefr ~ ., data=tr, method='svmLinear3', verbose=F)  # LibLineaR
        } else if (clf=='xgb') {
          mod <- train(cefr ~ ., data=tr, method='xgbLinear', verbose=F)  # XGBoost
        } else if (clf=='ker') {
          mod <- train(cefr ~ ., data=tr, method='mlpKerasDropout', verbose=F)  # Keras
        }
        # test set predictions with model
        print(paste('... testing on', test))
        preds <- predict(mod, newdata=te)
        fold.df <- cbind(fold.df, preds)  # add predictions to output file and name column with classifier
        colnames(fold.df)[ncol(fold.df)] <- clf
        # evaluate with classificationMetrics() from performanceEstimation
        cm <- data.frame(t(classificationMetrics(preds, trues=truth)))
        # prop within 1 level of the truth (sq to drop polarity)
        le1 <- sum((as.numeric(preds)-as.numeric(truth)^2)<=1) / length(truth)
        cm$propWithinOne <- le1
        # add expt info and append to results df
        end <- as.integer(Sys.time())
        dur <- end-start
        tim <- as.character(Sys.time())
        cm$run <- i
        cm$fold <- fld
        cm$classifier <- clf
	cm$pluslang <- pluslang
        cm$train <- train
        cm$test <- test
        cm$features <- features
        cm$UDversion <- udv
        cm$min.freq <- mindf
   	cm$min.freq.type <- 'doc.freq'
        cm$vocab_size <- vsize
        cm$durationSecs <- dur
        cm$timestamp <- tim
        eval.df <- rbind(eval.df, cm)
        # info print
        print(cm)
        print(paste('Language', lang, 'duration:', dur, 'sec. (', round(dur/60, 2), 'min.) @', tim))
      }
      # write fold info: append or not?
      if ((app==0) & (fld==1)) {
        write.table(fold.df, foldout, sep=',', row.names=F)
      } else {
        write.table(fold.df, foldout, sep=',', row.names=F, col.names=F, append=T)
      }
    }
  }
  print('========')
  print(paste('Language', lang, 'averaged metrics:'))
  eval.df %>% group_by(test, classifier) %>%
    summarize(P=mean(macroPrec, na.rm=T), R=mean(macroRec, na.rm=T), F=mean(macroF, na.rm=T), le1=mean(propWithinOne, na.rm=T)) %>% print()
  
  # save to file: append or not?
  if (app==0) {
    write.table(eval.df, fout, sep=',', row.names=F)
  } else {
    write.table(eval.df, fout, sep=',', row.names=F, col.names=F, append=T)
  }
}

