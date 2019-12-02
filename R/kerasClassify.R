## REPROLANG replication of Vajjala & Rama 2018, Experiments with Universal CEFR Classification, BEA.
## ACL Anthology: https://www.aclweb.org/anthology/W18-0515
## V&R's code repo: https://github.com/ nishkalavallabhi/UniversalCEFRScoring



## PRELIMS

# libraries: udpipe for parsing, tidytext for n-grams, magrittr for fwds pipe %>%
if (!require("pacman")) install.packages("pacman")
library(pacman)
pacman::p_load(tidyverse, tidytext, keras, performanceEstimation, nnet)



# function to prepare for NNs: transforming words to indices
text2int <- function(tdf, prefix, lang, minfreq, traintest, ttype) {
  print(paste('converting', ttype, 'to integers in the', traintest, 'set...'))
  vocabfile <- paste0(prefix, '/vocabs/', lang, '_', ttype, '.csv')
  tdf$tokens <- tolower(tdf$tokens)  # lower case all words
  # character or word tokenizer
  tok.df <- tdf %>% tidytext::unnest_tokens(output=token, input=tokens, token=ttype) %>% as_tibble()
  # if test: load vocab
  if (traintest=='test') {
    vocab <- as_tibble(read.csv(vocabfile, as.is=T))
  } else if (traintest=='train') {  # if train: create and store vocab
    # filter words for min frequency
    tok.counts <- tok.df %>% count(token, sort=T)
    vocab <- tok.counts %>% filter(n>=minfreq)  # gt as in V&R code
    vocab$idx <- 1:nrow(vocab) + 2  # token index with offset for pad, start, unk tokens
    vocab <- vocab %>% add_row(token='<PAD>', idx=0) %>% add_row(token='<START>', idx=1) %>% add_row(token='<UNK>', idx=2)
    vocab <- vocab %>% arrange(idx)
    # and write
    write.table(vocab, vocabfile, sep=',', row.names=F)
  }
  # max text length as in V&R code
  if (grepl('char', ttype)) {
    maxlen <- 2000
  } else {
    maxlen <- 400
  }
  # convert texts to integers, with padding up to max.length
  nrows <- nrow(tdf)
  padded <- matrix(nrow=nrows, ncol=maxlen)
  for (t in 1:nrows) {
    did <- tdf$doc.id[t]
    subs <- tok.df %>% filter(doc.id==did) %>% select(token)
    tokens <- subs$token
    tint <- 1  # initiate text integers with start token index
    for (tok in tokens) {
      if (length(tint)<maxlen) {  # ensure sequence does not exceed max.length (i.e. clip content)
        if (tok %in% vocab$token) {  # if a known word
          thistok <- vocab$idx[which(vocab$token==tok)]
        } else {
          thistok <- 2  # else oov
        }
        # append to text integers
        tint <- append(tint, thistok)
      }
    }
    # pad to max length and append to corpus texts
    if (length(tint)<maxlen) {
      tint <- c(tint, rep(0, maxlen-length(tint)))
    }
    padded[t,] <- tint
  }
  # add vocab size as final col
  vsize <- nrow(vocab)
  cbind(padded, rep(vsize, nrow(padded)))  # return
}


# function to train and test, store results
kerasEval <- function(in.df, train, test, mwf, fout, foldout, app, folds, prefix, pluslang, udv) {
  
  # data prep for training set
  print('========')
  clf <- 'keras'
  feats <- 'embeddings'
  lang <- train  # assume language=training.lang
  # drop unused CEFR levels
  in.df$cefr <- droplevels(in.df$cefr)
  # add test fold numbers
  in.df$testfold <- 0
  if (grepl('crossling', lang)) {  # set train as fold=0 and test as fold=1 for cross-ling expts
    train <- 'DE'
    print(paste('Cross-linguistic experiment on', train, 'train and', test, 'test'))
    in.df <- subset(in.df, lang %in% c(train, test))
    in.df$testfold[which(in.df$lang==test)] <- 1
  } else {  # else get test folds from pre-processed table: works for multiling and monoling expts
    print('Fetching test fold numbers for dataset...')
    for (r in 1:nrow(in.df)) {
      did <- in.df$doc.id[r]
      in.df$testfold[r] <- folds$testfold[which(folds$doc.id==did)]
    }
  }
  # start expt runs
  eval.df <- data.frame()
#  for (i in 1:1) {  # test: run thru once
  for (i in 1:10) {  # average 10 runs
    print(paste('Language', lang, ': start time for this run @', as.character(Sys.time())))
    for (fld in 1:max(in.df$testfold)) {  # for each fold
      print(paste('Run', i, 'fold', fld, 'for language', lang, 'preparing training data...'))
      # get train and test
      trsub <- subset(in.df, testfold!=fld)
      tesub <- subset(in.df, testfold==fld | testfold==-1)  # add any adversarial Spanish texts to test
      tr.word <- text2int(trsub, prefix, lang, mwf, 'train', 'words')  # convert word tokens to indices
      te.word <- text2int(tesub, prefix, lang, mwf, 'test', 'words')
      # get vocab size (in final col) and remove final column
      vsize <- as.numeric(tr.word[1,ncol(tr.word)])
      tr.word <- tr.word[,-ncol(tr.word)]
      te.word <- te.word[,-ncol(te.word)]
      # also character indices for multilingual expts
      csize <- 0
      if (grepl('multiling', lang)) {
        tr.char <- text2int(trsub, prefix, lang, 0, 'train', 'characters')  # convert characters to indices, min.freq=0 as in V&R
        te.char <- text2int(tesub, prefix, lang, 0, 'test', 'characters')
        # get vocab size (in final col) and remove final column
        csize <- as.numeric(tr.char[1,ncol(tr.char)])
        tr.char <- tr.char[,-ncol(tr.char)]
        te.char <- te.char[,-ncol(te.char)]
      }
      # CEFR labels for training, grab true CEFRs from test, convert to categorical binary class matrix (minus 1 b/c start at 0)
      cefr.levels <- levels(trsub$cefr)
      train.labs <- to_categorical(as.numeric(trsub$cefr)-1)
      n.levels <- length(cefr.levels)
      # language labels if aux.objective needed
      train.langs <- to_categorical(as.numeric(as.factor(trsub$lang))-1)
      n.langs <- length(unique(in.df$lang))
      truth <- tesub$cefr
      # prepare results df
      fold.df <- tesub %>% select(doc.id, testfold, lang, cefr)
      fold.df$min.freq <- mwf
      fold.df$min.freq.type <- 'word.freq'
      fold.df$run <- i
      fold.df$train <- train
      fold.df$test <- test
      fold.df$features <- feats
      fold.df$UDversion <- udv
      fold.df$vocab_size <- vsize
      fold.df$char_size <- csize
      fold.df$pluslang <- pluslang
      # we are go for training
      start <- as.integer(Sys.time())  # epoch seconds
      print('========')
      print(paste('Run', i, 'fold', fld, ': Keras classifier training on', train, 'language data...'))
      # Keras models for multiling and monoling expts
      if (grepl('multiling', lang)) {  # if multilingual experiments, add char embeddings
        # inputs and layers as in V&R vode
        main_input <- layer_input(shape=c(ncol(tr.word)), dtype='int32', name='word_input')
#        word_embed <- main_input %>% layer_embedding(input_dim=vsize, output_dim=32, input_length=ncol(tr.word)) %>%
        word_embed <- main_input %>% layer_embedding(input_dim=vsize, output_dim=200, input_length=ncol(tr.word)) %>%
          layer_spatial_dropout_1d(rate=0.25) %>% layer_flatten()
        aux_input <- layer_input(shape=c(ncol(tr.char)), dtype='int32', name='char_input')
#        char_embed <- aux_input %>% layer_embedding(input_dim=csize, output_dim=16, input_length=ncol(tr.char)) %>%
        char_embed <- aux_input %>% layer_embedding(input_dim=csize, output_dim=100, input_length=ncol(tr.char)) %>%
          layer_spatial_dropout_1d(rate=0.25) %>% layer_flatten()
        # if with language aux objective
        if (pluslang) {
          concat <- layer_concatenate(inputs=c(word_embed, char_embed))
          aux_output <- concat %>%
	    layer_dense(units=n.langs, activation='softmax', name='lang_output')
          main_output <- layer_concatenate(inputs=c(concat, aux_output)) %>%
            layer_dropout(rate=0.25) %>%
	    layer_dense(units=n.levels, activation='softmax', name='cefr_output')
          # define model: not sequential as in V&R
          model <- keras_model(inputs=c(main_input, aux_input), outputs=c(main_output, aux_output))
          # loss weights as in V&R
#          model %>% compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=list('accuracy'), loss_weights=c(1, .5))
          model %>% compile(loss='categorical_crossentropy', optimizer='adam', metrics=list('accuracy'), loss_weights=c(1, .5))
          # do training
          history <- model %>% fit(x=list(word_input=tr.word, char_input=tr.char), y=list(cefr_output=train.labs, lang_output=train.langs),
            epochs=8, batch_size=128, verbose=1)  # 8 epochs, batch size 128 as in V&R
          # test set predictions with trained model
          print(paste('... testing on', test))
          pred.mat <- model %>% predict(x=list(word_input=te.word, char_input=te.char))  # prediction prob matrix
          pred.mat <- pred.mat[[1]]  # level predictions only
        } else {  # without lang aux objective
          main_output <- layer_concatenate(inputs=c(word_embed, char_embed)) %>%
            layer_dropout(rate=0.25) %>%
	    layer_dense(units=n.levels, activation='softmax')
          model <- keras_model(inputs=c(main_input, aux_input), outputs=main_output)
#          model %>% compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=list('accuracy'))
          model %>% compile(loss='categorical_crossentropy', optimizer='adam', metrics=list('accuracy'))
          history <- model %>% fit(x=list(word_input=tr.word, char_input=tr.char), y=train.labs, epochs=8, batch_size=128, verbose=1)
          # test set predictions with trained model
          print(paste('... testing on', test))
          pred.mat <- model %>% predict(x=list(word_input=te.word, char_input=te.char))  # prediction prob matrix
        }
      } else {  # else a monolingual expt
        model <- keras_model_sequential()
        model %>%
#          layer_embedding(input_dim=vsize, output_dim=100, input_length=ncol(tr.word)) %>%  # 100 dim embedding as in V&R
          layer_embedding(input_dim=vsize, output_dim=300, input_length=ncol(tr.word)) %>%
	  layer_flatten() %>%
	  layer_dense(units=n.levels, activation='softmax')
#        model %>% compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=list('accuracy'))
        model %>% compile(loss='categorical_crossentropy', optimizer='adam', metrics=list('accuracy'))
        history <- model %>% fit(tr.word, train.labs, epochs=10, batch_size=32, verbose=1)  # 10 epochs, batch size 32 as in V&R
        # test set predictions with trained model
        print(paste('... testing on', test))
        pred.mat <- model %>% predict(te.word)  # prediction prob matrix
      }
      # evaluate predictions against ground truth
      pred.max <- apply(pred.mat, 1, function(row) which.is.max(row))  # get max per doc
      preds <- as.factor(cefr.levels[pred.max])  # convert predictions to CEFR levels
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
      cm$features <- feats
      cm$min.freq <- mwf
      cm$min.freq.type <- 'word.freq'
      cm$vocab_size <- vsize
      cm$char_size <- csize
      cm$durationSecs <- dur
      cm$timestamp <- tim
      eval.df <- rbind(eval.df, cm)
      # info print
      print(cm)
      print(paste('Run', i, 'fold', fld, 'language', lang, 'duration:', dur, 'sec. (', round(dur/60, 2), 'min.) @', tim))
      # write fold info: append or not?
      if ((app==0) & (fld==1) & (i==1)) {  # only if append=0, 1st fold, 1st run
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

