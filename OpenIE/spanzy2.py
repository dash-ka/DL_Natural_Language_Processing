import spacy
import sys
import numpy as np
import spacy.tokens
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from spacy.tokens import Span # a Span object from the slice doc[start : end].

nlp = spacy.load('en_core_web_lg')

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


def bottom_up(T):
    Q, R, L = [], [], []
    # record the number of children as the node attribute
    for n in T.nodes:
        T.nodes[n]["ch"] = len(T[n])
        
    # initialize the R queue with the root node
    root = [n for n in T.nodes if T.nodes[n]["dep_label"]=="ROOT"][0]
    R.append(root)
    
    # use R as auxiliary queue to search for leaves
    # descend the tree top-down collecting leaves in Q
    while len(R) > 0:
        v = R.pop(0)
        for child in T[v]:
            # if the node is a leaf, enqueue in Q
            if T.nodes[child]["ch"] == 0:
                Q.append(child)
            else:
                R.append(child) 
    num = 0
    while len(Q) > 0:
        v = Q.pop(0) # dequeue the front node
        L.append(v)
        num += 1
        T.nodes[v]["order"] = num
        if v != root: # if node is not ROOT
            parent = list(T.predecessors(v))[0]
            # decrease the number of children by 1
            T.nodes[parent]["ch"] = T.nodes[parent]["ch"]-1
            # append the parent once the list of its children has been exhausted
            if T.nodes[parent]["ch"] == 0:
                Q.append(parent)   
    return L

def top_down(T):
    Q, L = [], []
    root = [n for n in T.nodes if T.nodes[n]["dep_label"]=="ROOT"][0]
    Q.append(root)
    while len(Q)>0:
        v = Q.pop(0)
        L.append(v)
        for child in T[v]:
            Q.append(child)
    return L


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




def draw_graph(G, ax=None):
    
    # Tree Layout
    for v in preorder(G):
        if G.nodes[v]["head_id"] == G.nodes[v]["parent"]:
            G.nodes[v]["depth"] = 0
        else:
            parent = list(G.predecessors(v))[0]
            G.nodes[v]["depth"] = G.nodes[parent]["depth"] + 1
                
    for n in G.nodes:
        G.nodes[n]["breadth"] = 0
            
    for v in bottom_up(G):
        
        # if v is a leaf node
        if len(G[v]) == 0:
            G.nodes[v]["breadth"] = 1
        if G.nodes[v]["head_id"] != G.nodes[v]["parent"]:
            parent = list(G.predecessors(v))[0]
            G.nodes[parent]["breadth"] = G.nodes[parent]["breadth"] + G.nodes[v]["breadth"]
            
    sibling = 0
    for v in top_down(G):
        G.nodes[v]["y"] = - G.nodes[v]["depth"]
        if G.nodes[v]["head_id"] == G.nodes[v]["parent"]:
            G.nodes[v]["x"] = 0
        else:
            parent = list(G.predecessors(v))[0]
            brothers = list(child for child in G[parent])
            if v == brothers[0]:
                G.nodes[v]["x"] = G.nodes[parent]["x"] 
                sibling  = v
            
            else:
                G.nodes[v]["x"] = G.nodes[sibling]["x"] + G.nodes[sibling]["breadth"] 
                sibling = v
                
    for v in bottom_up(G):
        children = list(G[v])
        if len(children)>0:
            G.nodes[v]["x"] = (G.nodes[children[-1]]["x"] + G.nodes[children[0]]["x"])/2 
    
    labels = nx.get_node_attributes(G, 'node_text') 
    edge_labels = dict()
    for src, trg, data in G.edges(data=True):
        edge_labels[(src, trg)] = data["text"]
    
    pos_verbs = dict()
    pos_nouns = dict()
    label_verbs, label_nouns = dict(), dict()
    for i, d in G.nodes(data=True):
        if d["pos"] in ["VERB", "AUX"]:
            pos_verbs[i]= np.array([G.nodes[i]["x"], G.nodes[i]["y"]])
            label_verbs[i] = G.nodes[i]["node_text"]
        else:
            pos_nouns[i]= np.array([G.nodes[i]["x"], G.nodes[i]["y"]])
            label_nouns[i] = G.nodes[i]["node_text"]
        
    # styling and drawing
    bb = dict(boxstyle="round, pad=0.3", fc="w", ec="green", alpha=0.5, mutation_scale=10)
    nx.draw_networkx_edges(G, {**pos_nouns, **pos_verbs}, width=1.5, alpha=0.3, edge_color='#085554', ax=ax, connectionstyle="arc3")
    nx.draw_networkx_edge_labels(G, {**pos_nouns, **pos_verbs}, edge_labels=edge_labels, label_pos=0.5, font_size=13, bbox=bb, ax=ax, rotate=False)
    nx.draw_networkx_nodes(G, pos_verbs,nodelist=list(pos_verbs.keys()), node_size=5000, node_color = 'orange', alpha=0.3, ax=ax) 
    nx.draw_networkx_labels(G, pos_verbs, labels = label_verbs, font_color='k', alpha=0.9, font_size=17, font_family='sans-serif', ax=ax)
    nx.draw_networkx_nodes(G, pos_nouns, nodelist=list(pos_nouns.keys()), node_size=3500, node_color = '#085554', alpha=0.3, ax=ax) 
    nx.draw_networkx_labels(G, pos_nouns,  labels = label_nouns, font_color='k', alpha=0.9, font_size=17, font_family='sans-serif', ax=ax)
    
    # Resize figure for label readibility
    if ax is not None:
        ax.margins(x = 0.30, y = 0.10)
        ax.set_axis_off()
    else:
        plt.axis("off")





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
                parent_cluster[token.i]=parent_cluster[token.head.i]
                
                # UPDATE <scr> node's dictionary:
                src_node = list(token.children)[0]
                if src_node.i in skipped_ids:
                    for grandchild in src_node.children:
                        if grandchild.i in nodi:
                            nodi[grandchild.i]["parent"] = token.head.i
                if src_node.i in nodi:
                    nodi[src_node.i]["parent"]=parent_cluster[token.head.i]
                    nodi[src_node.i]["links"]={"type":"prep", 
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
                parent_cluster[token.i] = parent_cluster[token.head.i]
                
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
    if collapse_of:
        for node in to_collapse_:
            if node.i not in skipped_ids and node.i in nodi:
                
                skipped_ids.append(node.i)
                
                # collapse with parent
                parent = parent_cluster[nodi[node.i]["parent"]]
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


def find_valid_children(predicate, G):
    
    """Find all dependent nodes of the predicate that are either nominal phrases or subordinate clauses.
    
    Args:
        predicate(int): predicate node_index we want to find the triple object for
        G(networkx.Graph): restructured syntax tree of the sentence
        
    Returns:
        children(set[int]): ids of dependent nodes
        
    """
    
    children = set()
    for child in nx.descendants_at_distance(G, predicate, 1):

        is_nominal = lambda child: G.nodes[child]["pos"] in ["NOUN", "PROPN", "PRON", "DET", "ADJ"]  and G.nodes[child]["dep_label"] not in ["nsubj"]
        is_sub_clause = lambda child: G.nodes[child]["dep_label"] in ["ccomp", "xcomp", "pcomp", "pobj", "expl"] 
        
        if is_nominal(child) or is_sub_clause(child) :
            children = children.union([child])
            
    return children
    

def triples(G, extend_subject=True):
    
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
            
        # create prepositional triples    
        elif data["pos"] in ["NOUN", "DET", "PROPN", "PRON", "NUM"] and data["links"]:
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
    for v, data in verbs:#[::-1]:
        
        # [CASE 1]: search for <subj> among predicate children
        for child in nx.descendants_at_distance(G, v, 1):
            if G.nodes[child]["dep_label"] in ["nsubj", "nsubjpass", "csubj", "npadvmod"] :
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
        valid_children = find_valid_children(v, G) - {subj}
        
        # [OPTIONAL] Extending <subj> entity with dependent nominal phrase 
        if extend_subject:
            grandchildren = []
            for g in nx.descendants_at_distance(G, subj, 1):
                if G.nodes[g]["dep_label"] in ["dobj", "iobj", "pobj", "attr"]:
                    grandchildren.append(g)
                    
            subj = [subj, grandchildren[0]] if grandchildren else [subj]
            
        # [OPTIONAL] Extending <csubj> entity with dependent nominal <subj>
        elif subj in verb_subj_dict and verb_subj_dict[subj] != verb_subj_dict[v]:
            subj = [verb_subj_dict[subj], subj] 
        else:
            subj = [subj]
                
        # find <obj> for each predicate in the sentence 
        for child in valid_children:
            
            obj = [child]
            
            # if fold_preps = TRUE: join prepositions with predicate label
            predicate_suffix = ""
            
            # extract the preposition string from syntactic dependency label
            if G.nodes[child]["links"]:
                predicate_suffix = G.edges[G.nodes[child]["parent"], child]["text"]  
                        
            # join the predicate string label with its prepositional suffix (if any)
            # predicate = " ".join([G.nodes[v]["node_text"], predicate_suffix] )
                           
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