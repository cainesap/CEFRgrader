## REPROLANG replication of Vajjala & Rama 2018, Experiments with Universal CEFR Classification, BEA.
## ACL Anthology: https://www.aclweb.org/anthology/W18-0515
## V&R's code repo: https://github.com/ nishkalavallabhi/UniversalCEFRScoring


## libraries: udpipe for parsing, tidytext for n-grams, magrittr for fwds pipe %>%
if (!require("pacman")) install.packages("pacman", repos="https://cloud.r-project.org")
library(pacman)
pacman::p_load(udpipe, tidytext, magrittr, dplyr, caret)


## arg vars ##
args <- commandArgs(trailingOnly=T)
udpath <- args[1]
langtoolpath <- args[2]
indir <- args[3]
prefix <- args[4]
udv <- '2.0'
## end ##



# filepaths
fileout <- paste0(prefix, '/merlin_features_udpipe', udv, '.tsv')
textfile <- paste0(prefix, '/merlin_texts.tsv')
foldfile <- paste0(prefix, '/merlin_folds.tsv')


## selected UDPipe models for Czech, German, Italian: n.b. UD 2.0 models used in V&R paper, 2.4 latest available at present
## others are available, see https://universaldependencies.org
ud.cz <- udpipe_load_model(paste0(udpath, 'czech-ud-2.0-170801.udpipe'))
ud.de <- udpipe_load_model(paste0(udpath, 'german-ud-2.0-170801.udpipe'))
ud.it <- udpipe_load_model(paste0(udpath, 'italian-ud-2.0-170801.udpipe'))
#ud.cz <- udpipe_load_model(paste0(udpath, 'czech-pdt-ud-2.4-190531.udpipe'))
#ud.de <- udpipe_load_model(paste0(udpath, 'german-gsd-ud-2.4-190531.udpipe'))
#ud.it <- udpipe_load_model(paste0(udpath, 'italian-isdt-ud-2.4-190531.udpipe'))


## n-gram feature extractor, uses unnest_tokens() from tidytext, also count() and distinct() from dplyr
ngrams <- function(text, gram, n, doc.id) {
  txt.df <- data.frame(doc.id, text)
  gram.df <- txt.df %>% unnest_tokens(input=text, output=feats, token='ngrams', n=n)  # table of ngrams
  count.df <- gram.df %>% count(feats, sort=T, name='value')  # count of ngrams
  out.df <- distinct(merge(gram.df, count.df, by='feats'), feats, .keep_all=T)  # merge and cut to unique ngrams
  out.df$type <- paste0(gram, n)  # e.g. word3, pos5
  out.df  # returns df of n-grams, doc.id, count, n-gram type (add lang and cefr in main loop)
}


## feature extraction for each text in each language
## see Vajjala & Rama 2018 for feature descriptions (§3.2)
doc.id <- 0
totalFiles <- 0
doc.df <- data.frame()
langs <- c('CZ', 'DE', 'IT')
for (lang in langs) {
  files <- list.files(path=paste0(indir, lang), pattern='.txt', full.names=T)
  totalFiles <- totalFiles + length(files)
  for (filein in files) {
    doc.id <- doc.id + 1
    feat.df <- data.frame()
    
    # get level
    cefr <- gsub('.*_([ABC][12])\\.txt', '\\1', filein)
    
    # add to df of docs
    lineout <- data.frame(doc.id, lang, cefr)
    doc.df <- rbind(doc.df, lineout)
    
    # info print
    print(paste(lang, 'file', doc.id, 'of', totalFiles, 'at CEFR', cefr))
    
    # get text
    txt <- enc2utf8(readLines(filein))  # UTF-8 encoding
    txt <- gsub('"', "'", txt)  # replace double quotes with single quotes
    
    ## (0) pre-processing: tokenize, POS tag, dependency parse with UDPipe
    if (lang=='CZ') {
      ud.out <- udpipe_annotate(ud.cz, x=txt)
    } else if (lang=='DE') {
      ud.out <- udpipe_annotate(ud.de, x=txt)
    } else if (lang=='IT') {
      ud.out <- udpipe_annotate(ud.it, x=txt)
    }
    ud.df <- as.data.frame(ud.out)
    
    ## (1) word and pos n-grams (n from 1 to 5, excl n-grams with freq <10)
    tokens <- paste(ud.df$token, collapse=' ')  # use tokenized text
    for (n in 1:5) {
      feat.df <- rbind(feat.df, ngrams(tokens, 'word', n, doc.id))
    }
    postoks <- paste(ud.df$upos, collapse=' ')  # Universal POS tags
    for (n in 1:5) {
      feat.df <- rbind(feat.df, ngrams(postoks, 'pos', n, doc.id))
    }
    feat.df$lang <- lang
    feat.df$cefr <- cefr
    
    ## (2) save tokens for word and character embeddings
    nTokens <- nrow(ud.df)
    text.df <- data.frame(doc.id, lang, cefr, tokens, nTokens)
    if (doc.id==1) {  # 1st file with headers
      write.table(text.df, file=textfile, row.names=F, sep='\t')
    } else {
      write.table(text.df, file=textfile, row.names=F, col.names=F, sep='\t', append=T)
    }
    
    ## (3) dependency unigrams: relation, POS, head POS triples
    headtags <- unlist(lapply(ud.df$head_token_id, function(x) { x <- as.numeric(x); ifelse(x==0, 'ROOT', ud.df$upos[x]) } ))
    feats <- paste0(ud.df$dep_rel, '_', ud.df$upos, '_', headtags)
    type <- 'dep.triples'
    new.df <- data.frame(table(feats))  # count of dep.triples for this doc
    value <- new.df$Freq
    new.df <- cbind(new.df, doc.id, value, type, lang, cefr)
    new.df <- new.df[,-2]  # drop Freq col
    feat.df <- rbind(feat.df, new.df)
    
    ## (4a) doc length = n.tokens
    value <- nrow(ud.df)
    feats <- type <- 'doc.length'
    new.df <- data.frame(feats, doc.id, value, type, lang, cefr)
    feat.df <- rbind(feat.df, new.df)
    
    ## (4b) lexical density, variation, diversity
    # see Lu 2010: http://onlinelibrary.wiley.com/doi/10.1111/j.1540-4781.2011.01232_1.x/epdf
    # where n.lex OR uniq.lex = all OR unique open-class category words in UD POS (ADJ, ADV, INTJ, NOUN, PROPN, VERB)
    # lex density = n.lex/n.words
    # lex variation = uniq.lex/n.lex
    # lex diversity (TTR) = uniq.words/n.words
    ## filter punctuation
    no.punct <- subset(ud.df, upos!='PUNCT')
    lex.words <- subset(no.punct, upos %in% c('ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB'))$token
    n.lex <- length(lex.words)
    uniq.lex <- length(unique(lex.words))
    n.words <- nrow(no.punct)
    uniq.words <- length(unique(no.punct$token))
    value <- n.lex / n.words
    feats <- type <- 'lex.density'
    new.df <- data.frame(feats, doc.id, value, type, lang, cefr)
    feat.df <- rbind(feat.df, new.df)
    value <- uniq.lex / n.lex
    feats <- type <- 'lex.variation'
    new.df <- data.frame(feats, doc.id, value, type, lang, cefr)
    feat.df <- rbind(feat.df, new.df)
    value <- uniq.words / n.words
    feats <- type <- 'lex.diversity'
    new.df <- data.frame(feats, doc.id, value, type, lang, cefr)
    feat.df <- rbind(feat.df, new.df)
    
    ## (4c) spelling and grammar error counts from Language Tool (Czech unavailable)
    # see https://github.com/languagetool-org/languagetool
    tmpfile <- '~/tmp/reprolang.txt'
    grammerr <- 0
    spellerr <- 0
    if (lang!='CZ') {
      txt <- gsub('[^\\.]$', '.', txt)  # ensure final full-stop (otherwise language tool throws an error)
      writeLines(txt, tmpfile)
      langcode <- tolower(lang)
      if (lang=='DE') {  # German has spellchecker for German variant (append uppercase DE)
        langcode <- paste0(langcode, '-', lang)
      }
      spellgramm <- system(paste('java -jar', paste0(langtoolpath, '/languagetool-commandline.jar'), '-l', langcode, tmpfile), intern=T)
      spellgramm <- spellgramm[grepl('[0-9]+\\.\\)', spellgramm)]  # the error rules only
      spellgramm <- spellgramm[!grepl('WHITESPACE', spellgramm)]  # no whitespace rules
      spellgramm <- gsub('Line | column ', '', spellgramm)  # strip down to indices only
      memo <- c()
      for (sg in spellgramm) {
        loc <- gsub('.+([0-9]+,[0-9]+).+', '\\1', sg)
        if (!loc %in% memo) {  # only count 1 error per location
          grammerr <- grammerr + 1
          memo <- append(memo, loc)
          if (grepl('SPELL', sg)) {
            spellerr <- spellerr + 1
          }
        }
      }
    }
    value <- grammerr
    feats <- type <- 'gramm.error'
    new.df <- data.frame(feats, doc.id, value, type, lang, cefr)
    feat.df <- rbind(feat.df, new.df)
    value <- spellerr
    feats <- type <- 'spell.error'
    new.df <- data.frame(feats, doc.id, value, type, lang, cefr)
    feat.df <- rbind(feat.df, new.df)
    
    # info print
    print(paste('... extracted', nrow(feat.df), 'features from this file'))
    
    # and save
    if (doc.id==1) {  # 1st file with headers
      write.table(feat.df, file=fileout, row.names=F, sep='\t')
    } else {
      write.table(feat.df, file=fileout, row.names=F, col.names=F, sep='\t', append=T)
    }
  }
}

# create test fold numbers: stratified sampling with caret targeting CEFR level
print('Specifying test fold numbers...')
doc.df$testfold <- 0
out.df <- data.frame()
for (lg in langs) {
  subs <- subset(doc.df, lang==lg)
  subs$testfold <- caret::createFolds(subs$cefr, k=10, list=F)
  out.df <- rbind(out.df, subs)
}
write.table(out.df, file=foldfile, row.names=F, sep='\t')
print(paste('Saved to', fileout, textfile, foldfile))
