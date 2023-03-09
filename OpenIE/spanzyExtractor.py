import spacy
from collections import defaultdict
from spacy.tokens import Span


def resolve_coref(subj, predicate, G, verb_subj_dict=None):
    """Resolve coreference for <subj> entity.
    
    Args:
        subj(int): index of the subject entity to resolve
        predicate(int): index of the predicate
        G(networkx.Graph): restructured syntactic tree of the sentence
        verb_subj_dict(dict): dictionary of <predicate><subj> tuples
    Returns:
        subj(int): index of the resolved subject entity 
        verb_subj_dict(dict): dictionary of <predicate><subj> tuples
    
    """
    
    predicate_governor = G.nodes[predicate]["parent"]
            
    # resolved <subj> is the governor of the relative clause
    if G.nodes[predicate]["dep_label"] in ["relcl", "acl"]:
        subj = predicate_governor 
        
    if verb_subj_dict is None:
        return subj
        
    # 
    elif G.nodes[predicate]["dep_label"] in ["conj", "ccomp", "xcomp", "advcl", "pcomp"]:
        # resolved <subj> of the subordinate predicate is the <subj> of the governor predicate
        if G.nodes[predicate_governor]["dep_label"]!="xcomp" and predicate_governor in verb_subj_dict:
            subj = verb_subj_dict[predicate_governor]
            
        # heuristics: take the first nominal dependent node to resolve <subj>
        else:
            children = [child for child in G.nodes[predicate_governor]["children"] if \
                             G.nodes[child]["token_"].pos_ in ["NOUN", "PROPN"]]
            subj = children[0] if children else subj
                
    # overwrite the resolved <subj> 
    if G.nodes[subj]["token_"].pos_ in ["NOUN", "PROPN"]:
        verb_subj_dict[predicate] = subj 
            
    return subj, verb_subj_dict

def find_valid_children(predicate, G):
    
    """Find all dependent nodes of the predicate that are either nominal phrases or subordinate clauses.
    
    Args:
        predicate(int): predicate node_index we want to find the triple object for
        G(networkx.Graph): restructured syntax tree of the sentence
        
    Returns:
        children(set[int]): ids of dependent nodes
        
    """
    
    children = set()
    for child in G.nodes[predicate]["children"]:
                
        is_nominal = lambda child: G.nodes[child]["token_"].pos_ in ["NOUN", "PROPN", "PRON", "DET", "ADJ"] 
        is_sub_clause = lambda child: G.nodes[child]["dep_label"].split()[0] in ["ccomp", "xcomp", "pcomp", "pobj", "expl"] 
        
        if is_nominal(child) or is_sub_clause(child) :
            children = children.union([child])
            
    return children
    

def triples(G):
    
    """Extract structured triples of the form <subj><predicate><obj> from the restructured syntactic tree.
    
    Args: 
        G(networkx.Graph): restructured syntax tree of the sentence
    Returns:
        tripletes_dict(Dict[str, List]): a dictionary with list of (subject, object) tuples for each predicate key
    
    """
    
    tripletes_dict = defaultdict(list)
    verb_subj_dict = dict() 
    verbs = []
    
    
    for n, data in G.nodes(data=True):
        
        # collect verbal predicates
        if data["token_"].pos_ in["AUX", "VERB"]:
            verbs.append((n, data))
            
        # create an appositive proposition of the form <subj-"sameAS"-obj>
        elif data["dep_label"] in ["appos"]:
            tripletes_dict["sameAS"].append(([data["parent"]], [n])) 
            
        # create an existential proposition, for "there is"/"there are" constructions
        # will lead to the following proposition : <attr> is/are <there>
        elif data["dep_label"] in ["expl"]:
            attr_child = [child for child in G.nodes[data["parent"]]["children"] if \
                          G.nodes[child]["dep_label"] in ["attr"]]
            if attr_child:
                verb_subj_dict[data["parent"]] = attr_child[0]
                
                
 
    # find <subj> for each verbal predicate in a sentence 
    for v, data in verbs[::-1]:
        # [CASE 1]: search for <subj> among predicate children
        for child in data["children"]:
            if G.nodes[child]["dep_label"] in ["nsubj", "nsubjpass", "csubj"] :
                verb_subj_dict[v] = child
                
        # special case for "xcomp": <subj> is controlled by its syntactic governor
        if data["dep_label"] in ["xcomp"]: 
            governor_obj = [g for g in G.nodes[G.nodes[v]["parent"]]["children"]\
                             if G.nodes[g]["dep_label"] in ["dobj"]]
            if governor_obj:
                verb_subj_dict[v] = governor_obj[0]
        
        # if <subj> not found among children:
        if v not in verb_subj_dict:
            
            # [CASE 2]: subordinate or coordinate clause predicate inherits <subj> from its governor predicate 
            # LAST UPDATE: add "pcomp" to the dep list will extend the extraction list
            if data["dep_label"].split()[0] in ["conj", "ccomp", "xcomp", "pcomp", "advcl"] :
                if data["parent"] in verb_subj_dict:
                    verb_subj_dict[v] = verb_subj_dict[data["parent"]]
                        
            # [CASE 3]: <subj> is the nominal syntactic governor of the predicate
            elif data["parent"] != "ROOT" and \
                G.nodes[data["parent"]]["token_"].pos_ in ["NOUN", "PROPN", "PRON", "DET"]:
                verb_subj_dict[v] = data["parent"]
                
                
    # make an auxiliary dict for <subj> coreference resolution
    coref_verb_subj = verb_subj_dict.copy()
    
    for v, subj in verb_subj_dict.items():
        
        valid_children = find_valid_children(v, G) - {subj}
        
        # [OPTIONAL] Resolving co-reference for <subj>:
        if G.nodes[subj]["token_"].pos_ in ["PRON", "DET"] and G.nodes[v]["dep_label"] != "ROOT":
            subj, coref_verb_subj = resolve_coref(subj, v, G, coref_verb_subj)
            
        # [OPTIONAL] Extending <subj> entity with dependent nominal phrase 
        grandchildren = [g for g in G.nodes[subj]["children"] if \
                        G.nodes[g]["dep_label"].split()[0] in ["dobj", "iobj", "pobj"]]
        
        # [OPTIONAL] Extending <csubj> entity with dependent nominal <subj>
        if subj in verb_subj_dict and verb_subj_dict[subj] != verb_subj_dict[v]:
            subj = [verb_subj_dict[subj], subj] 
        else:
            subj = [subj]
                
        if grandchildren: 
            subj = subj + [grandchildren[0]]
    
            
        # [OPTIONAL] Extending <subj> entity with dependent nominal phrase (prepositional phrases)
        #grandchildren = [g for g in G.nodes[subj]["children"] if \
        #                     G.nodes[g]["dep_label"].split()[0] in ["dobj", "iobj", "pobj"]]
        #if grandchildren and G.nodes[subj]["token_"].pos_ in ["NOUN", "PROPN", "DET", "PRON"]:
        #    subj = [subj, grandchildren[0]]
        #else:
        #    subj = [subj]
        
        
        # find <obj> for each predicate in the sentence 
        for child in valid_children:
            
            obj = [child]
            
            # if fold_preps = TRUE: join prepositions with predicate label
            predicate_suffix = ""
            
            # extract the preposition string from syntactic dependency label
            if len(G.nodes[child]["dep_label"].split()) == 2:
                predicate_suffix = " " + G.nodes[child]["dep_label"].split()[-1] + " "
                        
            # join the predicate string label with its prepositional suffix (if any)
            predicate = G.nodes[v]["node_text"] + predicate_suffix 
            
            
            # ---------------------- Extending <obj> entity ----------------------------
                           
            # Extend subordinate clause <obj> with clause <subj> if <subj> != <subj> of v
            if child in verb_subj_dict and verb_subj_dict[child] != verb_subj_dict[v]:
                obj = [verb_subj_dict[child], child] 
                
            # [OPTIONAL] Extending <obj> entity with dependent nominal phrase 
            grandchildren = [g for g in G.nodes[child]["children"] if \
                             G.nodes[g]["dep_label"].split()[0] in ["dobj", "iobj", "pobj"]]
            
            if grandchildren: #(G.nodes[child]["dep_label"].split()[0] in ["pcomp"] or \
             #G.nodes[child]["token_"].pos_ in ["NOUN", "PROPN", "DET", "PRON"]):
                
                # record  <subj> <predicate> <obj> triple ???
                #tripletes_dict[predicate].append((subj, obj))
                
                for grandchild in grandchildren:
                    # record  <subj> <predicate> <obj> triple 
                    tripletes_dict[predicate].append((subj, obj + [grandchild]))
                    
            else:    
                # record  <subj> <predicate> <obj> triple 
                tripletes_dict[predicate].append((subj, obj))
                
    return tripletes_dict


def modified_annotate(triples_dict, nlp_sentence, G):
    
    """ Annotate a sentence with entity spans extracted from <subj><predicate><obj> triples.
    
    Args:
        triples_dict (Dict[str, List]): a dictionary with list of (subject, object) tuples for each predicate key
        nlp_sentence (Doc object): spacy sentence to annotate 
    
    Returns:
        nlp_sentence (Doc object): spacy sentence annotated with entities
    """
    
    ents = dict() # dictionary collecting entity spans
    annotated_nodes = set() # auxiliary set
    
    for predicate, subj_obj_tuples in triples_dict.items():
        for subj_nodes_ids, obj_nodes_ids in subj_obj_tuples:
          
            idx_subj, idx_obj = [], []
            
            for subj_node in subj_nodes_ids:
                # filter for <subj> entity components with specific pos tags
                if G.nodes[subj_node]["token_"].pos_ in ["NOUN", "PROPN", "PRON", "INTJ"]:
                    idx_subj += G.nodes[subj_node]["token_"]._.idx_list
   
            if idx_subj:
                idx_subj = sorted(idx_subj)
                bos, eos = idx_subj[0], idx_subj[-1]+1
                
                # create <subj> entity span
                if not set(range(bos, eos)).intersection(annotated_nodes):
                    ents[bos] = Span(nlp_sentence, bos, eos, label="ENTITY")
                    annotated_nodes = annotated_nodes.union(range(bos, eos))
                    
            for obj_node in obj_nodes_ids:
                # filter for <obj> entity components with specific pos tags
                if G.nodes[obj_node]["token_"].pos_ in ["NOUN", "PROPN"]:
                    idx_obj += G.nodes[obj_node]["token_"]._.idx_list
        
            if idx_obj:
                idx_obj = sorted(idx_obj)
                bos, eos = idx_obj[0], idx_obj[-1]+1
                
                # create <obj> entity span
                if not set(range(bos, eos)).intersection(annotated_nodes):
                    ents[bos] = Span(nlp_sentence, bos, eos, label="ENTITY")
                    annotated_nodes = annotated_nodes.union(range(bos, eos))
                    
    # set extracted entity spans at the document level, preserving spacy ner_spans (if missed by the algorithm)
    nlp_sentence.set_ents(ents.values(), default="unmodified")

    return nlp_sentence