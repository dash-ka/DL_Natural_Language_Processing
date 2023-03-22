""" Usage:
    spanzy_extract --in=INPUT_FILE --out=OUTPUT_FILE
"""
#from oie_readers.spanzyExtractor import *
from oie_readers.SpanIE import *
nlp = spacy.load("en_core_web_lg")
from docopt import docopt
import re

if __name__ == '__main__':

    args = docopt(__doc__)
    print(args)
    file_read = args['--in']
    file_write = args['--out']
    count = 0
    with open(file_read, "r", encoding="utf-8") as fin:
        for line in fin:
            sentence = line.strip()
            g = generate_graph(nlp(sentence), fold_preps=True, collapse_of=True)
            triples_dict = triples(g, extend_extractions=False, resolve_coreferences=True)
            fout = open(file_write, "a")


            for verb, subj_obj_list in triples_dict.items():
                if isinstance(verb, int):
                    for (subj, obj, suffix) in subj_obj_list:
                        arg1 = " ".join(g.nodes[s]["node_text"] for s in subj)
                        arg2 = suffix + " " + " ".join(g.nodes[o]["node_text"] for o in obj)
                        predicate = g.nodes[verb]["node_text"]
                        fout.write('{}\t{}\t{}\t{}\n'.format(sentence, predicate, arg1, arg2))
                        count += 1
    fout.close()

    print("Number of extractions over 3200 sentences:", count)


