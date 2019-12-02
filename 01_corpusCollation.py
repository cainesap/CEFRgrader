'''
Create a new dataset with new filenames, and removed metadata. 
Renamed corpora are in the Datasets/ folder in this repo
Original corpora for all languages can be downloaded from MERLIN website
http://www.merlin-platform.eu/C_data.php

AC additional: after corpus download, unzip merlin-text-v1.1, and 
point to $MERLIN/meta_ltext/ where $MERLIN is your location for the corpus
'''

import os, sys, re

if len(sys.argv) > 1:
    merlin = sys.argv[1] + "/"  # e.g. ~/Corpora/MERLIN/meta_ltext/
    outdir = sys.argv[2] + "/"  # e.g. ~/Corpora/MERLIN/text_only/
else:
    print("No argument variables supplied: please specify paths to MERLIN corpus and output directory")
    exit

#dirpath = "/Users/sowmya/Research/CrossLing-Scoring/Corpora/" #CZ_ltext_txt", DE_ltext_txt, IT_ltext_txt
#outputdirpath = "/Users/sowmya/Research/CrossLing-Scoring/Corpora/Renamed/"
#files = os.listdir(dirpath)
files = os.listdir(merlin)

#inputdirs = ["CZ_ltext_txt", "DE_ltext_txt", "IT_ltext_txt"]
inputdirs = ["czech", "german", "italian"]
outputdirs = ["CZ","DE","IT"]


# AC: replaced dirpath with merlin and outputdirpath with outdir below
print("Processing MERLIN...")
cefrpatt = re.compile('[A-C][12]')
cefrdict = {}
for i in range(0, len(inputdirs)):
    files = os.listdir(os.path.join(merlin,inputdirs[i]))
    lang = outputdirs[i]
    for file in files:
#        print(file)
        if file.endswith(".txt"):
            content = open(os.path.join(merlin,inputdirs[i],file),"r").read()
            text = content.split("Learner text:")[1].strip()
            cefr = content.split("Learner text:")[0].split("Overall CEFR rating: ")[1].split("\n")[0]
            if cefrpatt.match(cefr):  # if a valid CEFR level (i.e. ignore unscored texts)
                cefrkey = cefr + "_" + lang  # collect dictionary of file info
                if cefrkey in cefrdict:
                    cefrdict[cefrkey]['count'] += 1
                else:
                    cefrdict[cefrkey] = {}
                    cefrdict[cefrkey]['count'] = 1
                    cefrdict[cefrkey]['files'] = {}
                thiscount = cefrdict[cefrkey]['count']
                cefrdict[cefrkey]['files'][thiscount] = {}
                cefrdict[cefrkey]['files'][thiscount]['file'] = file
                cefrdict[cefrkey]['files'][thiscount]['lang'] = lang
                cefrdict[cefrkey]['files'][thiscount]['cefr'] = cefr
                cefrdict[cefrkey]['files'][thiscount]['text'] = text

# function to write texts
def text_write(filename, lang, cefr, text):
    newname = filename.replace(".txt","") + "_" + lang + "_" + cefr + ".txt"
    if not os.path.exists(outdir + lang):  # create directory if not found
        os.mkdir(outdir + lang)
    fh = open(os.path.join(outdir,lang,newname), "w")
    fh.write(text+"\n")
#    print("wrote: ", os.path.join(outdir,outputdirs[i],newname))

# threshold for inclusion?
# >=10 in paper
thresh = 10
for cefrkey in sorted(cefrdict):
    finalcount = cefrdict[cefrkey]['count']
    print("%i files found for %s" % (finalcount, cefrkey))
    if finalcount >= thresh:
        print("printing...")
        for filecount in cefrdict[cefrkey]['files']:
            file = cefrdict[cefrkey]['files'][filecount]['file']
            lang = cefrdict[cefrkey]['files'][filecount]['lang']
            cefr = cefrdict[cefrkey]['files'][filecount]['cefr']
            text = cefrdict[cefrkey]['files'][filecount]['text']
            text_write(file, lang, cefr, text)
    else:
        print("not printing...")
