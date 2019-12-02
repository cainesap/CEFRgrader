#!/bin/bash

# note prescribed directory structure
#$/ tree
#.
#├── input                   #directory containing all input data sets (actual data will be provided at runtime)
#└── output                  #directory containing output ouput data
#    ├── datasets            #directory for output data sets
#    └── tables_and_plots    #directory for output comparables, including scores, tables and/or plots,
#                            #etc. (i.e. the respective major reproduction comparables indicated in the call for papers)
#
####


# declare paths
MERLIN=input/meta_ltext/
TEXTS=input/text_only/
mkdir -p $MERLIN
mkdir -p $TEXTS
OUT=output/datasets/
LOGS=$OUT/logs/
mkdir -p $LOGS


# install UDPipe, download 2.4 models, install Language Tool
# wget from https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2998/udpipe-ud-2.4-190531.zip?sequence=2&isAllowed=y ?
# then unpack CZ DE IT
#ud.cz <- udpipe_load_model(paste0(udpath, 'czech-pdt-ud-2.4-190531.udpipe'))
#ud.de <- udpipe_load_model(paste0(udpath, 'german-gsd-ud-2.4-190531.udpipe'))
#ud.it <- udpipe_load_model(paste0(udpath, 'italian-isdt-ud-2.4-190531.udpipe'))
# or udpipe_download_model() in R?
UDPIPE=

# see https://github.com/languagetool-org/languagetool
# is it this? 
# curl -L https://raw.githubusercontent.com/languagetool-org/languagetool/master/install.sh | sudo bash <options>
LANGTOOL=

# R install pacman

# and run corpus prep with io argvars
python3 01_corpusCollation.py $MERLIN $TEXTS

# feature extraction
Rscript 02_featureExtraction.R $UDPIPE $LANGTOOL $TEXTS $OUT
# e.g.
udpath <- '~/workspace/nlp-tools/udpipe/'
langtoolpath <- '~/workspace/nlp-tools/LanguageTool-4.7-stable'

# experiments
Rscript 03_classificationExperiments_parallel_crossling.R $OUT
Rscript 03_classificationExperiments_parallel_monoling.R $OUT
Rscript 03_classificationExperiments_parallel_multiling.R $OUT

# print results
Rscript 04_resultsSummary.R $LOGS

