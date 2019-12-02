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
TABLES=output/tables_and_plots/


# install UDPipe, download 2.0 models
UDPIPE=udpipe/
mkdir $UDPIPE
cd $UDPIPE
# download 2.0 archive
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2364/udpipe-ud-2.0-170801.zip
unzip udpipe-ud-2.0-170801.zip
mv udpipe-ud-2.0-170801/czech-ud-2.0-170801.udpipe .  # CZ model
mv udpipe-ud-2.0-170801/german-ud-2.0-170801.udpipe .  # DE model
mv udpipe-ud-2.0-170801/italian-ud-2.0-170801.udpipe .  # IT model
# clean up
rm udpipe-ud-2.0-170801.zip
rm -r udpipe-ud-2.0-170801/


# install Language Tool
# see https://github.com/languagetool-org/languagetool
LANGTOOL=langtool/
curl -L https://raw.githubusercontent.com/languagetool-org/languagetool/master/install.sh | bash
mv LanguageTool-4.7-stable/ $LANGTOOL

# R install pacman
R -e "install.packages('pacman', repos='https://cloud.r-project.org')"


# REPROLANG 2020

# and run corpus prep with io argvars
python3 01_corpusCollation.py $MERLIN $TEXTS

# feature extraction
Rscript 02_featureExtraction.R $UDPIPE $LANGTOOL $TEXTS $OUT

# experiments
Rscript 03_classificationExperiments_parallel_crossling.R $OUT
Rscript 03_classificationExperiments_parallel_monoling.R $OUT
Rscript 03_classificationExperiments_parallel_multiling.R $OUT

# print results
Rscript 04_resultsSummary.R $LOGS $TABLES
