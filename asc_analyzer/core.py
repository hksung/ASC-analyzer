# ASC Analyzer
import spacy
from spacy.tokens import Doc
from spacy.language import Language
import json
import glob
import statistics as stat
import math
import os
import warnings
from collections import defaultdict
import pprint

# Suppress warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)
warnings.filterwarnings(
    "ignore",
    message="Model '.*' was trained with spaCy v3.7.4 and may not be 100% compatible",
    category=UserWarning
)

# Load pipelines
nlp = spacy.load("en_core_web_trf", exclude=["ner"])
model_dir = "/Users/hakyungsung/Documents/GitHub-tch/ASC-analyzer/asc_analyzer/models/asc_model"
ascNLP = spacy.load(model_dir)

def fullExtractSent(sent,verbose = False):
	doc = nlp(sent)
	docEnts = ascNLP(sent)
	sentList = []
	entD = {}
	for ent in docEnts.ents:
		entD[str(ent.start_char)] = ent.label_
		#print((str(ent.start_char), ent.text, ent.lemma_, ent.label_))
	#print(entD)
	for token in doc:
		tokL = [str(token.i +1),token.text,token.lemma_,token.pos_,token.tag_,str(token.morph), str(token.head.i + 1),token.dep_, "_"]
		if str(token.idx) not in entD:
			tokL.append("_")
		else:
			tokL.append(entD[str(token.idx)])
		sentList.append(tokL)
		if verbose == True:
			print(tokL)
		#print(token.i, token.idx,token.text,token.lemma_,token.pos_, token.dep_) #token.idx is start character
	return(sentList)

def fullExtractDoc(text,verbose = False):
	doc = nlp(text)
	docEnts = ascNLP(text)
	docList = []
	entD = {}
	for ent in docEnts.ents:
		entD[str(ent.start_char)] = ent.label_
		#print((str(ent.start_char), ent.text, ent.lemma_, ent.label_))
	#print(entD)
	for sent in doc.sents:
		sentList = []
		sentidx = 0
		w1idx = 0
		for token in sent:
			if sentidx == 0:
				w1idx = token.i
			sentidx += 1
			tokL = [str(sentidx),token.text,token.lemma_,token.pos_,token.tag_,str(token.morph), str(token.head.i - w1idx + 1),token.dep_, "_"]
			if str(token.idx) not in entD:
				tokL.append("_")
			else:
				tokL.append(entD[str(token.idx)])
			sentList.append(tokL)
			if verbose == True:
				print(tokL)
			#print(token.i, token.idx,token.text,token.lemma_,token.pos_, token.dep_) #token.idx is start character
		docList.append(sentList)
	return(docList)


def ascExtractDoc(text, ascFreqDict, ascSoaDict, verbose=False):
    doc = nlp(text)
    ents = {str(ent.start_char): ent.label_.replace("_", "-") for ent in ascNLP(text).ents}
    docs = []
    for sent in doc.sents:
        rows = []
        start = sent[0].i
        for idx, token in enumerate(sent, start=1):
            row = [str(idx), token.text, token.lemma_]
            label = ents.get(str(token.idx))
            print(label)
            if not label or token.lemma_ == '.':
                row.append("\t".join(["_"]*8))
            else:
                row.append(label)
                lemma_label = f"{token.lemma_}_{label}"
                # lemma freq
                row.append(
                    round(math.log(ascFreqDict['lemmaFreq'].get(token.lemma_,0)),3)
                    if token.lemma_ in ascFreqDict['lemmaFreq'] else "_"
                )
                # asc freq
                row.append(
                    round(math.log(ascFreqDict['ascFreqD'].get(label,0)),3)
                    if label in ascFreqDict['ascFreqD'] else "_"
                )
                # lemma-asc freq
                row.append(
                    round(math.log(ascFreqDict['ascLemmaFreqD'].get(lemma_label,0)),3)
                    if lemma_label in ascFreqDict['ascLemmaFreqD'] else "_"
                )
                # SOA metrics
                for m in ['mi','tscore','deltap_lemma_cue','deltap_structure_cue']:
                    row.append(
                        round(ascSoaDict[m].get(lemma_label,0),3)
                        if lemma_label in ascSoaDict[m] else "_"
                    )
            rows.append(row)
            if verbose:
                print(row)
        docs.append(rows)
    return docs


def conlluString(meta, conlluList):
    body = "\n".join(["\t".join(r) for r in conlluList])
    return f"{meta}\n{body}"


def processText(text):
	ascDict = {"lemmas" : [], "ascs" : [], "asc+lemmas" : []}
	processed = fullExtractDoc(text)
	for sent in processed:
		vacIdxList = []
		for token in sent:
			if token[9] != "_":
				ascDict["lemmas"].append(token[2])
				ascDict["ascs"].append("-".join(token[9].split("_")))
				ascDict["asc+lemmas"].append("_".join([token[2],"-".join(token[9].split("_"))]))
				vacIdxList.append(token[0]) #get list of idxs to process next
	return(ascDict)


def ttr(strList):
	if len(strList) == 0:
		return(0)
	else:
		return(len(set(strList))/len(strList))

def safe_divide(numerator, denominator):
	if denominator == 0 or denominator == 0.0:
		index = 0
	else: index = numerator/denominator
	return index

def MATTR(text, window_length = 11):
	vals = []
	windows = []
	if len(text) < (window_length + 1):
		index = safe_divide(len(set(text)),len(text))
		return(index)

	else:
		for x in range(len(text)):
			small_text = text[x:(x + window_length)]
			if len(small_text) < window_length:
				break
			windows.append(small_text)
			vals.append(safe_divide(len(set(small_text)),float(window_length)))
	index = stat.mean(vals)

	return(index)

def proportion(strList, target):
	if len(strList) == 0:
		return(0)
	else:
		return(strList.count(target)/len(strList))

def mvRefiner(tList,ignore = [None]):
	return([x for x in tList if x not in ignore])

def ascRefiner(tList, targetASC = [None], lemmaIgnore = [None]):
	refinedList = []
	for item in tList:
		items = item.split("_")
		if len(items) >=3:
			continue
		if len(items) < 2:
			print(item)
			continue
		lemma = items[0]
		asc = items[1]
		if lemma in lemmaIgnore:
			continue
		elif targetASC != None and asc not in targetASC:
			continue
		else:
			refinedList.append(item)
	return(refinedList)

def freqLookup(fDict,itemList,returnList = False, logged = True, cutoff = 5,ignore = [None]):
	outList = []
	for x in itemList:
		if x not in fDict:
			outList.append("n/a")
		elif x in ignore:
			outList.append("ignored")
		else:
			val = fDict[x]
			if val < cutoff:
				outList.append("n/a")
			else:
				if logged == True:
					outList.append(math.log(val))
				else:
					outList.append(val)
	filtered = [x for x in outList if x not in ["n/a","ignored"]]
	if len(filtered) == 0:
		outAv = 0
	else:
		outAv = sum(filtered)/len(filtered)
	if returnList == True:
		return(outList)
	else:
		return(outAv)

def soaLookup(fDict,itemList,returnList = False):
	outList = []
	for x in itemList:
		if x not in fDict:
			outList.append("n/a")
		else:
			outList.append(fDict[x])
	filtered = [x for x in outList if x not in ["n/a"]]
	if len(filtered) == 0:
		outAv = 0
	else:
		outAv = sum(filtered)/len(filtered)
	if returnList == True:
		return(outList)
	else:
		return(outAv)

def indexCalc(ascDict,freqD,ascD):
	indexDict = {} #finish this
	for x in ascDict: #add ascDict items to outputDict
		indexDict[x] = ascDict[x]

	#create no "be" versions of lists:
	indexDict["lemmasNoBe"] = mvRefiner(indexDict["lemmas"],["be"])
	#print(indexDict["lemmasNoBe"])
	indexDict["asc+lemmasNoBe"] = ascRefiner(indexDict["asc+lemmas"],None,lemmaIgnore = ["be"])

	#create specific ASC lists
	indexDict["asc+lemmas_TRAN-S"] = ascRefiner(indexDict["asc+lemmas"],targetASC = ["TRAN-S"])
	indexDict["asc+lemmas_ATTR"] = ascRefiner(indexDict["asc+lemmas"],targetASC = ["ATTR"])
	indexDict["asc+lemmas_INTRAN-S"] = ascRefiner(indexDict["asc+lemmas"],targetASC = ["INTRAN-S"])
	indexDict["asc+lemmas_PASSIVE"] = ascRefiner(indexDict["asc+lemmas"],targetASC = ["PASSIVE"])
	indexDict["asc+lemmas_INTRAN-MOT"] = ascRefiner(indexDict["asc+lemmas"],targetASC = ["INTRAN-MOT"])
	indexDict["asc+lemmas_TRAN-RES"] = ascRefiner(indexDict["asc+lemmas"],targetASC = ["TRAN-RES"])
	indexDict["asc+lemmas_CAUS-MOT"] = ascRefiner(indexDict["asc+lemmas"],targetASC = ["CAUS-MOT"])
	indexDict["asc+lemmas_DITRAN"] = ascRefiner(indexDict["asc+lemmas"],targetASC = ["DITRAN"])
	indexDict["asc+lemmas_INTRAN-RES"] = ascRefiner(indexDict["asc+lemmas"],targetASC = ["INTRAN-RES"])

	#simple indices
	indexDict["clauseCount"] = len(indexDict["lemmas"])
	indexDict["clauseCountNoBe"] = len(indexDict["lemmasNoBe"])

	indexDict["ascTTR"] = ttr(indexDict["ascs"])

	indexDict["ascLemmaTTR"] = ttr(indexDict["asc+lemmas"])
	indexDict["ascLemmaTTRNoBe"] = ttr(indexDict["asc+lemmasNoBe"])

	indexDict["ascMATTR11"] = MATTR(indexDict["ascs"])

	indexDict["ascLemmaMATTR11"] = MATTR(indexDict["asc+lemmas"])

	indexDict["ascLemmaMATTR11NoBe"] = MATTR(indexDict["asc+lemmasNoBe"])

	#asc proportion indices
	indexDict["TRAN-S_Prop"]  = proportion(indexDict["ascs"], "TRAN-S")
	indexDict["ATTR-Prop"]  = proportion(indexDict["ascs"], "ATTR")
	indexDict["INTRAN-S_Prop"]  = proportion(indexDict["ascs"], "INTRAN-S")
	indexDict["PASSIVE_Prop"]  = proportion(indexDict["ascs"], "PASSIVE")
	indexDict["INTRAN-MOT_Prop"]  = proportion(indexDict["ascs"], "INTRAN-MOT")
	indexDict["TRAN-RES_Prop"]  = proportion(indexDict["ascs"], "TRAN-RES")
	indexDict["CAUS-MOT_Prop"]  = proportion(indexDict["ascs"], "CAUS-MOT")
	indexDict["DITRAN_Prop"]  = proportion(indexDict["ascs"], "DITRAN")
	indexDict["INTRAN-RES_Prop"]  = proportion(indexDict["ascs"], "INTRAN-RES")

	#insert code for calculating frequency indices here.
	#freqlookup(fDict,itemList,returnList = False, logged = True, cutoff = 5)
	#freqlookup(fDict,itemList,returnList = False, logged = True, cutoff = 5)

	indexDict["ascAvFreq"] = freqLookup(freqD["ascFreqD"],indexDict["ascs"])
	indexDict["ascFreq"] = freqLookup(freqD["ascFreqD"],indexDict["ascs"],returnList = True)

	indexDict["ascLemmaAvFreq"] = freqLookup(freqD["ascLemmaFreqD"],indexDict["asc+lemmas"])
	indexDict["ascLemmaFreq"] = freqLookup(freqD["ascLemmaFreqD"],indexDict["asc+lemmas"],returnList = True)
	# asc soa here
	indexDict["ascAvMI"] = soaLookup(ascD["mi"],indexDict["asc+lemmas"])
	indexDict["ascAvTscore"] = soaLookup(ascD["tscore"],indexDict["asc+lemmas"])
	indexDict["ascAvDPLemmaCue"] = soaLookup(ascD["deltap_lemma_cue"],indexDict["asc+lemmas"])
	indexDict["ascAvDPStructureCue"] = soaLookup(ascD["deltap_structure_cue"],indexDict["asc+lemmas"])

	# Specific ASC SOA
	indexDict["TRAN-S_AvMI"]  = soaLookup(ascD["mi"],indexDict["asc+lemmas_TRAN-S"])
	#indexDict["ATTR_AvMI"]  = soaLookup(ascD["mi"],indexDict["asc+lemmas_ATTR"])
	indexDict["INTRAN-S_AvMI"]  = soaLookup(ascD["mi"],indexDict["asc+lemmas_INTRAN-S"])
	indexDict["PASSIVE_AvMI"]  = soaLookup(ascD["mi"],indexDict["asc+lemmas_PASSIVE"])
	indexDict["INTRAN-MOT_AvMI"]  = soaLookup(ascD["mi"],indexDict["asc+lemmas_INTRAN-MOT"])
	indexDict["TRAN-RES_AvMI"]  = soaLookup(ascD["mi"],indexDict["asc+lemmas_TRAN-RES"])
	indexDict["CAUS-MOT_AvMI"]  = soaLookup(ascD["mi"],indexDict["asc+lemmas_CAUS-MOT"])
	indexDict["DITRAN_AvMI"]  = soaLookup(ascD["mi"],indexDict["asc+lemmas_DITRAN"])
	indexDict["INTRAN-RES_AvMI"]  = soaLookup(ascD["mi"],indexDict["asc+lemmas_INTRAN-RES"])

	indexDict["TRAN-S_Tscore"]  = soaLookup(ascD["tscore"],indexDict["asc+lemmas_TRAN-S"])
	#indexDict["ATTR_Tscore"]  = soaLookup(ascD["tscore"],indexDict["asc+lemmas_ATTR"])
	indexDict["INTRAN-S_Tscore"]  = soaLookup(ascD["tscore"],indexDict["asc+lemmas_INTRAN-S"])
	indexDict["PASSIVE_Tscore"]  = soaLookup(ascD["tscore"],indexDict["asc+lemmas_PASSIVE"])
	indexDict["INTRAN-MOT_Tscore"]  = soaLookup(ascD["tscore"],indexDict["asc+lemmas_INTRAN-MOT"])
	indexDict["TRAN-RES_Tscore"]  = soaLookup(ascD["tscore"],indexDict["asc+lemmas_TRAN-RES"])
	indexDict["CAUS-MOT_Tscore"]  = soaLookup(ascD["tscore"],indexDict["asc+lemmas_CAUS-MOT"])
	indexDict["DITRAN_Tscore"]  = soaLookup(ascD["tscore"],indexDict["asc+lemmas_DITRAN"])
	indexDict["INTRAN-RES_Tscore"]  = soaLookup(ascD["tscore"],indexDict["asc+lemmas_INTRAN-RES"])

	indexDict["TRAN-S_DPLemmaCue"]  = soaLookup(ascD["deltap_lemma_cue"],indexDict["asc+lemmas_TRAN-S"])
	#indexDict["ATTR_"]  = soaLookup(ascD["deltap_lemma_cue"],indexDict["asc+lemmas_ATTR"])
	indexDict["INTRAN-S_DPLemmaCue"]  = soaLookup(ascD["deltap_lemma_cue"],indexDict["asc+lemmas_INTRAN-S"])
	indexDict["PASSIVE_DPLemmaCue"]  = soaLookup(ascD["deltap_lemma_cue"],indexDict["asc+lemmas_PASSIVE"])
	indexDict["INTRAN-MOT_DPLemmaCue"]  = soaLookup(ascD["deltap_lemma_cue"],indexDict["asc+lemmas_INTRAN-MOT"])
	indexDict["TRAN-RES_DPLemmaCue"]  = soaLookup(ascD["deltap_lemma_cue"],indexDict["asc+lemmas_TRAN-RES"])
	indexDict["CAUS-MOT_DPLemmaCue"]  = soaLookup(ascD["deltap_lemma_cue"],indexDict["asc+lemmas_CAUS-MOT"])
	indexDict["DITRAN_DPLemmaCue"]  = soaLookup(ascD["deltap_lemma_cue"],indexDict["asc+lemmas_DITRAN"])
	indexDict["INTRAN-RES_DPLemmaCue"]  = soaLookup(ascD["deltap_lemma_cue"],indexDict["asc+lemmas_INTRAN-RES"])

	indexDict["TRAN-S_DPStructureCue"]  = soaLookup(ascD["deltap_structure_cue"],indexDict["asc+lemmas_TRAN-S"])
	#indexDict["ATTR-Prop"]  = soaLookup(ascD["deltap_structure_cue"],indexDict["asc+lemmas_ATTR"])
	indexDict["INTRAN-S_DPStructureCue"]  = soaLookup(ascD["deltap_structure_cue"],indexDict["asc+lemmas_INTRAN-S"])
	indexDict["PASSIVE_DPStructureCue"]  = soaLookup(ascD["deltap_structure_cue"],indexDict["asc+lemmas_PASSIVE"])
	indexDict["INTRAN-MOT_DPStructureCue"]  = soaLookup(ascD["deltap_structure_cue"],indexDict["asc+lemmas_INTRAN-MOT"])
	indexDict["TRAN-RES_DPStructureCue"]  = soaLookup(ascD["deltap_structure_cue"],indexDict["asc+lemmas_TRAN-RES"])
	indexDict["CAUS-MOT_DPStructureCue"]  = soaLookup(ascD["deltap_structure_cue"],indexDict["asc+lemmas_CAUS-MOT"])
	indexDict["DITRAN_DPStructureCue"]  = soaLookup(ascD["deltap_structure_cue"],indexDict["asc+lemmas_DITRAN"])
	indexDict["INTRAN-RES_DPStructureCue"]  = soaLookup(ascD["deltap_structure_cue"],indexDict["asc+lemmas_INTRAN-RES"])

	#asc soa lists here
	indexDict["ascListMI"] = soaLookup(ascD["mi"],indexDict["asc+lemmas"],returnList = True)
	indexDict["ascListTscore"] = soaLookup(ascD["tscore"],indexDict["asc+lemmas"],returnList = True)
	indexDict["ascListDPLemmaCue"] = soaLookup(ascD["deltap_lemma_cue"],indexDict["asc+lemmas"],returnList = True)
	indexDict["ascListDPStructureCue"] = soaLookup(ascD["deltap_structure_cue"],indexDict["asc+lemmas"],returnList = True)

	return(indexDict)

def indexCalcFull(loFiles,freqD,ascD):
	outDict = {}
	nfiles = len(loFiles)
	counter = 1
	for fname in loFiles:
		simplefname = fname.split("/")[-1]
		print("Processing",simplefname,counter,"of",nfiles,"files")
		counter += 1
		text = open(fname, errors = "ignore").read().strip() # open text
		outDict[simplefname] = indexCalc(processText(text),freqD,ascD)
	return(outDict)

def writeCsv(fullDict, indexNames, outName):
    import csv

    with open(outName, "w", newline="") as outf:
        writer = csv.writer(outf)
        # header
        writer.writerow(["filename"] + indexNames)
        # rows
        for fname, metrics in fullDict.items():
            row = [fname] + [str(metrics[idx]) for idx in indexNames]
            writer.writerow(row)
    outf.close()
# Optional: retain ASC output functions

def writeASCoutput(newFname,docAscList,header=["idx","token","lemma","asc","lemmaFreq","ascFreq","ascLemmaFreq","mi","tscore","deltap_lemma_cue","deltap_structure_cue"]):
    with open(newFname,'w') as outf:
        outf.write('\t'.join(header)+'\n')
        for sent in docAscList:
            outf.write('\n'.join(['\t'.join(x) for x in sent])+'\n\n')


def processCorpusASC(indir,outdir,suffix,freqD,soaD):
    files=glob.glob(indir+'*'+suffix)
    for i,fn in enumerate(files,1):
        name=os.path.basename(fn)
        print(f"Processing: {name} ({i}/{len(files)})")
        txt=open(fn,errors='ignore').read()
        ascList=ascExtractDoc(txt,freqD,soaD)
        writeASCoutput(os.path.join(outdir,name.replace(suffix,'_ASCinfo'+suffix)),ascList)
    print(f"Processed {len(files)} files")
