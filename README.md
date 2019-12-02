# CEFRgrader

Code accompanying REPROLANG 2020 submission
This repository relates to the [REPROLANG 2020 shared task](https://lrec2020.lrec-conf.org/en/reprolang2020) @ the LREC conference,
 specifically Task D.2 replicating the following paper:

[Sowmya Vajjala & Taraka Rama, 2018. Experiments with Universal CEFR classifications. In Proceedings of BEA](https://www.aclweb.org/anthology/W18-0515). (V&R)

----

### REPROLANG 2020 Task D.2

_Andrew Caines, @cainesap, University of Cambridge, UK_

[V&R's original code](https://github.com/nishkalavallabhi/UniversalCEFRScoring) needed little amendment.
Please refer to that repository and readme for background information. This repository describes my replication efforts only.

The main issue related to lack of clarity about the workflow.
Here I list the steps to get from corpus download to end results. Any errors are my own.

#### Pre-processing

1. download the MERLIN corpus (all files) from CLARIN via https://merlin-platform.eu/C_data.php
2. unzip downloaded file, unzip `merlin-text-v1.1.zip` and move `merlin-text-v1.1/meta_ltext` to your preferred location (= $MERLIN)
3. make an output directory for processed files (= $OUTDIR)
4. download or clone this repository (i.e. `git clone https://github.com/cainesap/reprolang_github.git`)
5. change directory to root of the repository (i.e. `cd reprolang_github`)
6. run `python3 01_corpusCollation.py $MERLIN $OUTDIR`
7. note that the exclusion of files described in the V&R paper is now handled in step 6 (rather than posthoc removal with a separate script)

#### Feature extraction

8. install the `udpipe` R library (i.e. `> install.packages('udpipe')`) and download version 2.0 models (not the latest models; but to match the ones V&R used) for Czech, German and Italian from [LINDAT/CLARIN](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2364?show=full)
9. install the Language Tool from [source](https://github.com/languagetool-org/languagetool) (i.e. `curl -L https://raw.githubusercontent.com/languagetool-org/languagetool/master/install.sh | bash`; now your path to 'LanguageTool-V.v-stable' = $LANGTOOL
10. run `Rscript 02_featureExtraction.R $UDPIPE $LANGTOOL $INDIR $FEATSFILE` for feature extraction from texts, as well as test-fold definition through stratified sampling, and print out of tokenised text

#### Experiments

11. run `Rscript 03_classificationExperiments_[monoling|multiling|crossling].R` to run monolingual, multilingual, crosslingual experiments like V&R
12. run `Rscript 04_resultsSummary.R` to print a summary of results based on experiment logs


_Andrew Caines, December 2019_
