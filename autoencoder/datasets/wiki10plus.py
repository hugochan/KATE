'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import os
import re

from ..preprocessing.preprocessing import init_stopwords, tiny_tokenize_xml, get_all_files

pattern = r'>([^<>]+)<'
prog = re.compile(pattern)
cached_stop_words = init_stopwords()

def extract_contents(text, out_file):
    if not isinstance(text, unicode):
        text = text.decode('utf-8')
    contents = ' '.join(prog.findall(text))
    contents = tiny_tokenize_xml(contents, False, cached_stop_words)
    with open(out_file, 'w') as f:
        f.write(' '.join(contents))

    return contents

def xml2text(in_dir, out_dir, white_list=None):
    # it will be fast if white_list is a dict instead of a list
    files = get_all_files(in_dir, recursive=False)
    count = 0
    for filename in files:
        if white_list and not os.path.basename(filename) in white_list:
            continue
        try:
            with open(filename, 'r') as fp:
                text = fp.read().lower()
                extract_contents(text, os.path.join(out_dir, os.path.basename(filename)))
                count += 1
        except Exception as e:
            raise e
        if count % 500 == 0:
            print 'processed %s' % count

    print 'processed %s docs, discarded %s docs' % (count, len(files) - count)
    import pdb;pdb.set_trace()

