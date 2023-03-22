import spacy
import numpy as np
import spacy.tokens
import networkx as nx
from collections import defaultdict


def preorder(T):
    stack, visited = [], []
    num = 0
    root = [n for n in T.nodes if T.nodes[n]["dep_label"]=="ROOT"][0]
    stack.append(root)
    while len(stack)>0:
        num += 1
        v = stack.pop()
        T.nodes[v]["order"] = num
        visited.append(v)
        for child in list(T.successors(v))[::-1]:
            stack.append(child)
    return visited

def spacy_postorder(T):
    S, L = [], []
    root = [t for t in T if t.dep_ == "ROOT"][0]
    S.append(root)
    while len(S) > 0:
        v = S.pop()
        if v.dep_ == "ROOT":
            v._.depth = 0
        else:
            v._.depth = v.head._.depth + 1
        L.append(v)
        for child in list(v.children):
            S.append(child)
    return L[::-1]



def generate_graph(nlp_sentence, fold_preps=True, collapse_of=True, skip_tokens=None, noun_phrase=None, predicate=None):
        
    if noun_phrase is None:
        noun_phrase = ["det", "poss", "case", "nummod", "cls", "compound", "amod", "acomp", "nmod", "conj", "cc", "advmod", "prep", "neg", "quantmod"]
    if predicate is None:
        predicate = ["aux", "auxpass", "prt", "neg",  "agent", "dative","case","compound"] 
    if skip_tokens is None:
        skip_tokens = ["punct","mark"]
        
    spacy.tokens.Token.set_extension("depth", default=None, force=True)
    nlp_traversal = spacy_postorder(nlp_sentence)
    
    # set up necessary data structures:
    G = nx.DiGraph()
    nodi = dict()
    skipped_ids, to_collapse_ = [], []
    parent_cluster = list(range(len(nlp_traversal))) 
    
    for index, token in enumerate(nlp_traversal):
        
        # set up a new attributed description for a node
        node = {"head_id":         token.i, 
                "head_label":      token.text,
                "lemma":           token.lemma_,
                "pos":             token.pos_,
                "dep":             token.dep_,
                "collapsed_ids":   [],
                "collapsed_text":  [],
                "parent":          parent_cluster[token.head.i], 
                "links":           {}
                }
        
        # create a temporary instance for a new node or retrieve the existing node
        tmp_node = node if token.i not in nodi else nodi[token.i]
        
        # verify if the head node is a predicate
        head_is_predicate = token.head.pos_ in ["AUX", "VERB"] 
        
        #  --------------------------------------------------------------------------
        # SKIPPING:
        
        if token.dep_ in skip_tokens:
            skipped_ids.append(token.i)
            parent_cluster[token.i] = token.head.i
            continue
            
            
        #  --------------------------------------------------------------------------
        # FOLDING of preposition and discourse words with a single descendent:
        # these will cease to represent nodes, and will become attributes on the acr from <scr> to <parent> nodes
        
        if fold_preps and token.dep_ in ["prep", "agent", "dative", "mark"]:
            
            #  ----------------------- Prepositions ---------------------------
            if token.dep_ in ["prep", "agent", "dative"] and len(list(token.children))==1:
               
                # record the id of the folded node
                skipped_ids.append(token.i)
                
                # preposizione perde l'identità, dicendo "sono folded, non fate più rif a me!"
                parent_cluster[token.i] = parent_cluster[token.head.i]
                
                # UPDATE <scr> node's dictionary:
                src_node = list(token.children)[0]
                parent_cluster[src_node.i] = parent_cluster[token.head.i]
                
                if src_node.i in skipped_ids:
                    for grandchild in src_node.children:
                        parent_cluster[grandchild.i] = parent_cluster[token.head.i]
                        if grandchild.i in nodi:
                            nodi[grandchild.i]["parent"] = parent_cluster[token.head.i]
                            
                            
                elif src_node.i in nodi:
                    nodi[src_node.i]["parent"] = parent_cluster[token.head.i]
                    nodi[src_node.i]["links"] = {"type":"prep", 
                                               "token_id":token.i,
                                               "text":token.text}
                
                    
                # [OPTIONAL] mark src nodes preceded by "of" preposition to be collapsed later with the parent node
                if collapse_of and token.text in ["of"]:
                    to_collapse_.append(src_node)
                    
                    
            #  ----------------------- Discourse Words ---------------------------
            # UPDATE or CREATE a new node record 
            elif token.dep_ in ["mark"]:
                
                # record the id of the folded node and change its cluster
                skipped_ids.append(token.i)
                parent_cluster[token.i]=parent_cluster[token.head.i]
                
                if token.head.i in nodi:
                    nodi[token.head.i]["links"] = {"type":"mark", 
                                     "token_id":token.i,
                                     "text":token.text}
                else:
                    tmp_node["head_id"] = token.head.i
                    tmp_node["head_label"] = token.head.text
                    tmp_node["lemma"] = token.head.lemma_
                    tmp_node["pos"] = token.head.pos_
                    tmp_node["dep"] = token.head.dep_
                    tmp_node["collapsed_ids"] = []
                    tmp_node["collapsed_text"] = []
                    tmp_node["parent"] = parent_cluster[token.head.head.i]
                    tmp_node["links"] = {"type":"mark", 
                                     "token_id":token.i,
                                     "text":token.text}
                    # record the parent node
                    nodi[tmp_node["head_id"]] = tmp_node
            
            else:
                nodi[tmp_node["head_id"]] = tmp_node
                    
        #  --------------------------------------------------------------------------
        # COLLAPSING nominal and predicate components with their governor
        
        elif (token.dep_ in noun_phrase * (1-head_is_predicate)) \
            or (token.dep_ in predicate) * head_is_predicate:
          
            # avoid collapsing adjectival modifiers already registered as nodes
            if not ("amod" in token.dep_ and token.i in nodi):
                # record the id of the collapsed node
                skipped_ids.append(token.i)
                
                # Nodo collassato perde la sua identità, e si identifica nel nodo parent,
                # il nodo collassato non potrà essere usato come parent di altri nodi
                parent_cluster = [
                    parent_cluster[token.head.i] if parent_cluster[idx] == token.i else parent_cluster[idx] for idx in
                    range(len(parent_cluster))]

                # CASE 1: parent node already exists:
                # UPDATE the parent dictionary with info about the collapsed node
                if token.head.i in nodi:
                    
                    nodi[token.head.i]["collapsed_ids"].append(token.i)
                    nodi[token.head.i]["collapsed_text"].append(token.text)
                    
                    # CASE 2: both parent and scr node exist:
                    # transfer all ids and words from scr node to parent and remove the scr record
                    if token.i in nodi:
                        nodi[token.head.i]["collapsed_ids"].extend(nodi[token.i]["collapsed_ids"])
                        nodi[token.head.i]["collapsed_text"].extend(nodi[token.i]["collapsed_text"])
                        del nodi[token.i]
                        
                # CASE 3: neither scr nor parent nodes exist:
                # CREATE a new attributed record for the parent node,
                # including the info about the collapsed node (eventually delete scr if exists)
                else:
                    tmp_node["head_id"] = token.head.i
                    tmp_node["head_label"] = token.head.text
                    tmp_node["lemma"] = token.head.lemma_
                    tmp_node["pos"] = token.head.pos_
                    tmp_node["dep"] = token.head.dep_
                    tmp_node["collapsed_ids"].append(token.i)
                    tmp_node["collapsed_text"].append(token.text)
                    tmp_node["parent"] = parent_cluster[token.head.head.i] 
                    
                    # register the attributed node
                    nodi[tmp_node["head_id"]] = tmp_node
                    if token.i in nodi:
                        del nodi[token.i]                    
                    
                # acknoledge children of collapsed scr node about their new parent
                for child in list(token.children):
                    if child.i in skipped_ids:
                        for grandchild in child.children:
                            if grandchild.i in nodi:
                                nodi[grandchild.i]["parent"] = token.head.i
                    if child.i in nodi:
                        nodi[child.i]["parent"] = token.head.i          

                # change the pointer to the parent on collapsed nodes as well
                for collapsed_id in tmp_node["collapsed_ids"]:
                    parent_cluster[collapsed_id] = parent_cluster[token.i]

            else:
                nodi[tmp_node["head_id"]] = tmp_node
            
        # CREATE a new node record if not folded/skipped or collapsed
        elif token.i not in nodi:
            nodi[tmp_node["head_id"]] = tmp_node
            
         
    # merge scr and parent nodes that are separated by "of" preposition (arc attribute) in a single node
    try:

        if collapse_of:
            for node in to_collapse_:
                if node.i not in skipped_ids and node.i in nodi:
                
                    skipped_ids.append(node.i)
                
                    # collapse with parent
                    parent = parent_cluster[node.i]
                    nodi[parent]["collapsed_ids"].extend(
                        [node.i, nodi[node.i]["links"]["token_id"]] + nodi[node.i]["collapsed_ids"])
                    nodi[parent]["collapsed_text"].extend(
                        [node.text, nodi[node.i]["links"]["text"]] + nodi[node.i]["collapsed_text"])
            
                    parent_cluster[node.i] = parent
                
                
                    for child in node.children:
                        parent_cluster[child.i] = parent
                        if child.i in nodi:
                            nodi[child.i]["parent"] = parent
                        
                    for collapsed_id in nodi[node.i]["collapsed_ids"]:
                        parent_cluster[collapsed_id] = parent
                    
                    del nodi[node.i]

    except:
        print(nlp_sentence)

                        
    # GRAPH CONSTRUCTION
    for node_id, data in nodi.items():
        
        # CHECK on pointers!
        if data["parent"] in skipped_ids:
            nodi[node_id]["parent"] = parent_cluster[data["parent"]]
        
        # glue words on collapsed nodes into appropriate textual label
        sorted_indices = np.argsort(np.array(data["collapsed_ids"] + [node_id]))
        glued_tokens = " ".join(list(np.array(data["collapsed_text"] + [data["head_label"]])[sorted_indices]))
        
        # add attributed nodes to the graph 
        G.add_node(node_id,
                   head_id = node_id, 
                   head_label = data["head_label"],
                   lemma = data["lemma"],
                   pos = data["pos"], 
                   dep_label = data["dep"],
                   parent = data["parent"],
                   collapsed_ids = data["collapsed_ids"],
                   node_text = glued_tokens, 
                   links = data["links"]["text"] if data["links"].get("type") else "")
        
        # add attributed edges to the graph
        if node_id != data["parent"]: # skip for the "ROOT"
            prepositional_link = data.get("links", "")
            if prepositional_link:
                G.add_edge(data["parent"], node_id,  
                           label=prepositional_link["type"].lower(),
                           text=prepositional_link["text"].lower())
            else:
                G.add_edge(data["parent"], node_id,
                           label=data["dep"].lower(),
                           text=data["dep"].lower())
       
    # add children attribute
    nodes = list(G.nodes())
    for node_id in nodes:
        G.nodes[node_id]["children"] = nx.descendants_at_distance(G, node_id, 1)
        
    return G


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
        
    elif G.nodes[predicate]["dep_label"] in ["conj", "ccomp", "xcomp", "advcl", "pcomp"]:
        #  <subj> of the subordinate predicate is the <subj> of the governor predicate
        if G.nodes[predicate_governor]["dep_label"]!="xcomp" and predicate_governor in verb_subj_dict:
            subj = verb_subj_dict[predicate_governor]
            
        # heuristics: take the first nominal dependent node to resolve <subj>
        else:
            children = [child for child in nx.descendants_at_distance(G, predicate_governor, 1) if \
                             G.nodes[child]["pos"] in ["NOUN", "PROPN"]]
            subj = children[0] if children else subj
                
    # overwrite the resolved <subj> 
    if G.nodes[subj]["pos"] in ["NOUN", "PROPN"]:
        verb_subj_dict[predicate] = subj 
            
    return subj, verb_subj_dict


def find_children(predicate, G):
    
    """Find all dependent nodes of the predicate that are either nominal phrases or subordinate clauses.
    
    Args:
        predicate(int): predicate node_index we want to find the triple object for
        G(networkx.Graph): restructured syntax tree of the sentence
        
    Returns:
        children(set[int]): ids of dependent nodes
        
    """
    POS_NOMINAL =  ["NOUN", "PROPN", "PRON", "DET", "ADJ"]
    DEP_CLAUSE = ["ccomp", "xcomp", "pcomp", "pobj", "expl"] 
    is_nominal = lambda child: G.nodes[child]["pos"] in POS_NOMINAL and \
                               G.nodes[child]["dep_label"] not in ["nsubj"]
    is_clause = lambda child: G.nodes[child]["dep_label"] in DEP_CLAUSE
       
    children = set()
    for child in nx.descendants_at_distance(G, predicate, 1):
        if is_nominal(child) or is_clause(child) :
            children = children.union([child])
            
    return children
    

def triples(G, extend_extractions=True, extend_subject=True, resolve_coreferences=True):
    
    """Extract structured triples of the form <subj><predicate><obj> from the restructured syntactic tree.
    
    Args: 
        G(networkx.Graph): restructured syntax tree of the sentence
        
    Returns:
        tripletes_dict(Dict[str, List]): a dictionary with list of (subject, object) tuples for each predicate key
    
    """
    
    tripletes_dict = defaultdict(list)
    verb_subj_dict = dict() 
    verbs = []
    
    for n in preorder(G):
        data = G.nodes[n]
        
        # collect verbal predicates
        if data["pos"] in["AUX", "VERB"]:
            verbs.append((n, data))
            
        # extract non-verbal relational tuples  
        if extend_extractions:
            
            if data["pos"] in ["NOUN", "DET", "PROPN", "PRON", "NUM"] and data["links"]:
                if G.nodes[data["parent"]]["pos"] in ["NOUN", "DET", "PROPN", "PRON", "NUM"]:
                    tripletes_dict[data["links"]].append(([data["parent"]], [n], ""))
                    
            # create an appositive proposition of the form <subj-"sameAS"-obj>
            elif data["dep_label"] in ["appos"]:
                tripletes_dict["sameAS"].append(([data["parent"]], [n], "")) 
            
            # create an existential proposition, for "there is"/"there are" constructions: <attr> is/are <there>
            elif data["dep_label"] in ["expl"]:
                attr_child = [child for child in nx.descendants_at_distance(G, data["parent"], 1) if \
                          G.nodes[child]["dep_label"] in ["attr"]]
                if attr_child:
                    verb_subj_dict[data["parent"]] = attr_child[0]
                    
 
    # find <subj> for each verbal predicate in a sentence 
    for v, data in verbs:
        
        # [CASE 1]: search for <subj> among predicate children
        for child in nx.descendants_at_distance(G, v, 1):
            if G.nodes[child]["dep_label"] in ["nsubj", "nsubjpass", "csubj"] :
                verb_subj_dict[v] = child
                
        # special case for "xcomp": <subj> is controlled by its syntactic governor
        if data["dep_label"] in ["xcomp"]: 
            governor_obj = [g for g in nx.descendants_at_distance(G, data["parent"], 1)\
                             if G.nodes[g]["dep_label"] in ["dobj"]]
            if governor_obj:
                verb_subj_dict[v] = governor_obj[0]
        
        # if <subj> not found among children:
        if v not in verb_subj_dict:
            
            # [CASE 2]: subordinate or coordinate clause predicate inherits <subj> from its governor predicate 
            # LAST UPDATE: add "pcomp" to the dep list will extend the extraction list
            if data["dep_label"] in ["conj", "ccomp", "xcomp", "pcomp", "advcl"] :
                if data["parent"] in verb_subj_dict:
                    verb_subj_dict[v] = verb_subj_dict[data["parent"]]
                        
            # [CASE 3]: <subj> is the nominal syntactic governor of the predicate
            elif data["parent"] != "ROOT" and \
                G.nodes[data["parent"]]["pos"] in ["NOUN", "PROPN", "PRON", "DET"]:
                verb_subj_dict[v] = data["parent"]
                

    # make an auxiliary dict for <subj> coreference resolution
    coref_verb_subj = verb_subj_dict.copy()
    
    for v, subj in verb_subj_dict.items():
        
        valid_children = find_children(v, G) - {subj}
        
        # [OPTIONAL] Extending <subj> entity with dependent nominal phrase 
        if extend_subject:
            grandchildren = []
            for g in nx.descendants_at_distance(G, subj, 1):
                if G.nodes[g]["dep_label"] in ["dobj", "iobj", "pobj", "attr"]:
                    grandchildren.append(g)
                    
            subj = [subj, grandchildren[0]] if grandchildren else [subj]
            
        # Extending <csubj> entity with dependent nominal <subj>
        elif subj in verb_subj_dict and verb_subj_dict[subj] != verb_subj_dict[v]:
            subj = [verb_subj_dict[subj], subj] 
        else:
            subj = [subj]
            
            
        # ------------------------------------------    
        if resolve_coreferences:
            if G.nodes[subj[-1]]["pos"] in ["DET", "PRON"] and G.nodes[v]["dep_label"] != "ROOT":
                new_subj, coref_verb_subj = resolve_coref(subj[-1], v, G, coref_verb_subj)
                subj[-1] = new_subj

            
        # find <obj> for each predicate in the sentence 
        for child in valid_children:
            
            obj = [child]
            
            # if fold_preps = TRUE: join prepositions with predicate label            
            # extract the prepositions from syntactic dependency label
            predicate_suffix = ""
            if G.nodes[child]["links"]:
                predicate_suffix = G.edges[G.nodes[child]["parent"], child]["text"]
            tripletes_dict[v].append((subj, obj, predicate_suffix))
            
          
    # ---------------------- Extending <obj> entity ----------------------------
    extended_tripletes_dict = tripletes_dict.copy()
    for v, pairs in tripletes_dict.items():
        if isinstance(v, int):
            
            extended_tripletes_dict[v] = []
            for (subj, obj, suffix) in pairs:
                               
                for (s, o, suff) in tripletes_dict.get(obj[0], ""):
                    if subj[-1]!=s[-1]:
                        extended_tripletes_dict[v].append((subj, s+obj+o, suffix))
                        break
                else:
                    extended_tripletes_dict[v].append((subj, obj, suffix))
                    
    return extended_tripletes_dict