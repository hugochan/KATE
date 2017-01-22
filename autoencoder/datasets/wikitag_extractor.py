'''
Created on Jan, 2017

@author: hugo

Note: The original code is from https://github.com/ravitejatv/IREMajorProjectDocClassification/blob/master/lda/tag_extractor.py
'''
from __future__ import absolute_import
import xml.sax
from collections import Counter

tags = []             #list of xml tags
labels = []           #list of all labels
labeldict = {}        #hash to labels mapping
titledict = {}        #hash to title mapping

class XMLhandler(xml.sax.ContentHandler):
    def __init__(self):
        self.tags=[]            #list of all tags
        self.articlenumber = 0
        self.content = ""
        self.articlehash = ""
        self.hash = ""
    def startElement(self,name,attrs):      #runs at start of all tags (name)
        if name not in self.tags:
            self.tags.append(name)
            tags.append(name)
            self.content = ""
    def endElement(self,name):
        if name == "hash":
            self.hash=self.content.strip()
            labeldict[self.hash] = []
        if name == "name":
            labels.append(self.content.strip())
            labeldict[self.hash].append(self.content.strip())
        if name == "title":
            titledict[self.hash] = self.content.strip()
        self.content = ""

    def characters(self,content):
        w = content.encode('utf-8').strip()
        if w > 0:
            self.content += w + "\n"


def extract_labels(in_file, topn):
    #XML PARSING
    parser = xml.sax.make_parser()
    parser.setContentHandler(XMLhandler())
    parser.parse(in_file)

    #NOTE in the xml file "css"  and "files" also come in the <hash>....</hash>, so i remove them manually
    labeldict.pop("css")
    labeldict.pop("files")

    #TO TAKE LABELS WITH AT LEAST N OCCURENCES ONLY
    freqs = Counter(labels)
    pairs = sorted(freqs.items(), key=lambda item: item[1], reverse=True)
    pairs = pairs[:topn]
    newlabels = zip(*pairs)[0]
    newlabeldict = {}

    #REMOVE EXTRA LABELS FROM LABEL DICTIONARY (HASH-LABEL MAPPING)
    #ALSO CHECK IF WE LOST ANY DOCUMENT; IN CASE A DOCUMENT HAS 0 LABELS

    se = set(newlabels)
    for i in labeldict:
        newlabeldict[i] = list(se.intersection(labeldict[i]))
        if len(newlabeldict[i]) == 0:
            newlabeldict.pop(i)

    return newlabeldict
