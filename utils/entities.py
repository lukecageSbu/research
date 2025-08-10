"""
Named Entity Recognition and overlap utilities with advanced NLP techniques.
Includes Stanford Stanza, spaCy, and enhanced rule-based methods.
"""

import re
import spacy
import stanza
from typing import List, Set, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import warnings
import torch

# Global NLP models - loaded once for efficiency
_nlp_models = {
    'spacy': None,
    'stanza': None
}

def load_stanza_model(lang: str = "en", processors: str = "tokenize,pos,lemma,ner,depparse,coref") -> stanza.Pipeline:
    """
    Load Stanford Stanza model with caching.
    
    Args:
        lang: Language code (default: "en")
        processors: Comma-separated list of processors to use
        
    Returns:
        nlp: Loaded Stanza pipeline
    """
    global _nlp_models
    
    if _nlp_models['stanza'] is None:
        try:
            # Try to download models if not present
            try:
                stanza.download(lang, verbose=False)
            except:
                pass  # Models might already be downloaded
                
            _nlp_models['stanza'] = stanza.Pipeline(
                lang=lang,
                processors=processors,
                use_gpu=torch.cuda.is_available(),
                verbose=False
            )
            print(f"✅ Loaded Stanford Stanza with processors: {processors}")
        except Exception as e:
            warnings.warn(f"Stanford Stanza model '{lang}' failed to load: {e}")
            warnings.warn("Falling back to spaCy/rule-based entity extraction.")
            _nlp_models['stanza'] = None
    
    return _nlp_models['stanza']


def load_spacy_model(model_name: str = "en_core_web_sm") -> spacy.Language:
    """
    Load spaCy model with caching.
    
    Args:
        model_name: Name of spaCy model to load
        
    Returns:
        nlp: Loaded spaCy model
    """
    global _nlp_models
    
    if _nlp_models['spacy'] is None:
        try:
            _nlp_models['spacy'] = spacy.load(model_name)
            # Disable unnecessary pipeline components for speed
            _nlp_models['spacy'].disable_pipes(["tagger", "parser", "lemmatizer"])
        except OSError:
            warnings.warn(f"spaCy model '{model_name}' not found. Install with: python -m spacy download {model_name}")
            warnings.warn("Falling back to rule-based entity extraction.")
            _nlp_models['spacy'] = None
    
    return _nlp_models['spacy']


def extract_entities_stanza(text: str) -> Dict[str, Set[str]]:
    """
    Extract entities using Stanford Stanza with advanced features.
    
    Args:
        text: Input text string
        
    Returns:
        entity_info: Dictionary with different types of extracted information
    """
    nlp = load_stanza_model()
    
    if nlp is None:
        # Fallback to spaCy
        return {"entities": extract_entities_spacy(text), "mentions": set(), "concepts": set()}
    
    # Process text with Stanza
    doc = nlp(text)
    
    entities = set()
    mentions = set()  # Coreference mentions
    concepts = set()  # Dependency-based concepts
    
    # 1. Extract named entities
    for sentence in doc.sentences:
        for entity in sentence.ents:
            if entity.type in {"PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART", "LAW", 
                              "LANGUAGE", "PRODUCT", "NORP", "FAC", "LOC"} and len(entity.text.strip()) > 2:
                # Normalize entity text
                entity_text = entity.text.strip().lower()
                entity_text = re.sub(r'^(the|a|an)\s+', '', entity_text)
                entity_text = re.sub(r'\s+(inc|ltd|corp|co)\.?$', '', entity_text)
                if len(entity_text) > 2:
                    entities.add(entity_text)
    
    # 2. Extract coreference mentions (if available)
    if hasattr(doc, 'coref_chains') and doc.coref_chains:
        for chain in doc.coref_chains:
            # Get the representative mention (usually the most complete form)
            representative = None
            max_length = 0
            
            for mention in chain.mentions:
                mention_text = mention.text.strip().lower()
                if len(mention_text) > max_length:
                    max_length = len(mention_text)
                    representative = mention_text
            
            if representative and len(representative) > 2:
                mentions.add(representative)
                # Add all mentions in the chain
                for mention in chain.mentions:
                    mention_text = mention.text.strip().lower()
                    if len(mention_text) > 2:
                        mentions.add(mention_text)
    
    # 3. Extract important concepts using dependency parsing
    for sentence in doc.sentences:
        for word in sentence.words:
            # Look for subjects, objects, and important modifiers
            if word.deprel in {'nsubj', 'dobj', 'iobj', 'nmod', 'compound'} and word.pos in {'NOUN', 'PROPN'}:
                if len(word.text) > 3 and not word.text.lower() in {'this', 'that', 'these', 'those'}:
                    concepts.add(word.text.lower())
            
            # Extract compound nouns
            if word.deprel == 'compound':
                head_word = sentence.words[word.head - 1] if word.head > 0 else None
                if head_word and head_word.pos in {'NOUN', 'PROPN'}:
                    compound = f"{word.text} {head_word.text}".lower()
                    concepts.add(compound)
    
    return {
        "entities": entities,
        "mentions": mentions,
        "concepts": concepts
    }


def extract_entities_spacy(text: str) -> Set[str]:
    """
    Extract named entities using spaCy NER.
    
    Args:
        text: Input text string
        
    Returns:
        entities: Set of extracted entity strings (normalized)
    """
    nlp = load_spacy_model()
    
    if nlp is None:
        # Fallback to rule-based extraction
        return extract_entities_rules(text)
    
    # Process text with spaCy
    doc = nlp(text)
    
    entities = set()
    for ent in doc.ents:
        # Filter out unwanted entity types and short entities
        if (ent.label_ in {"PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART", "LAW", 
                          "LANGUAGE", "PRODUCT", "NORP", "FAC"} and 
            len(ent.text.strip()) > 2):
            # Normalize entity text
            entity_text = ent.text.strip().lower()
            # Remove common prefixes/suffixes
            entity_text = re.sub(r'^(the|a|an)\s+', '', entity_text)
            entity_text = re.sub(r'\s+(inc|ltd|corp|co)\.?$', '', entity_text)
            if len(entity_text) > 2:
                entities.add(entity_text)
    
    return entities


def extract_entities_rules(text: str) -> Set[str]:
    """
    Extract named entities using improved rule-based patterns.
    
    Args:
        text: Input text string
        
    Returns:
        entities: Set of extracted entity strings
    """
    entities = set()
    
    # Pattern 1: Capitalized words/phrases (improved)
    capitalized_patterns = [
        r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b',  # Multi-word proper nouns
        r'\b[A-Z][a-z]+(?:\'s)?\b',  # Single proper nouns with possessive
        r'\b[A-Z]{2,}\b',  # Acronyms
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Two-word proper nouns
    ]
    
    for pattern in capitalized_patterns:
        matches = re.findall(pattern, text)
        entities.update(matches)
    
    # Pattern 2: Quoted entities
    quoted_entities = re.findall(r'"([^"]{3,})"', text)
    entities.update(quoted_entities)
    
    # Pattern 3: Date patterns (years, specific formats)
    date_patterns = [
        r'\b(19|20)\d{2}\b',  # Years
        r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # Date formats
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        if isinstance(matches[0], tuple) if matches else False:
            matches = [''.join(match) for match in matches]
        entities.update(matches)
    
    # Pattern 4: Title + Name patterns
    title_patterns = [
        r'\b(?:Mr|Mrs|Ms|Dr|Prof|President|Director|CEO|Chairman)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
    ]
    
    for pattern in title_patterns:
        matches = re.findall(pattern, text)
        entities.update(matches)
    
    # Filter out common stopwords and short entities
    enhanced_stopwords = {
        'the', 'this', 'that', 'these', 'those', 'a', 'an', 'and', 'or', 'but',
        'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'as', 'of',
        'it', 'he', 'she', 'they', 'we', 'you', 'i', 'his', 'her', 'their',
        'when', 'where', 'why', 'what', 'how', 'who', 'which', 'whose',
        'was', 'were', 'is', 'are', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'also', 'one', 'two', 'three', 'first', 'second', 'third', 'new', 'old',
        'year', 'years', 'time', 'times', 'way', 'ways', 'many', 'much', 'more',
        'most', 'some', 'all', 'any', 'each', 'every', 'both', 'either', 'neither'
    }
    
    filtered_entities = set()
    for entity in entities:
        # Clean and normalize
        clean_entity = re.sub(r'[^\w\s]', '', entity.lower().strip())
        
        # Skip if too short, is stopword, or is purely numeric
        if (len(clean_entity) > 2 and 
            clean_entity not in enhanced_stopwords and
            not clean_entity.isdigit() and
            not re.match(r'^\d+[a-z]*$', clean_entity)):
            filtered_entities.add(clean_entity)
    
    return filtered_entities


def extract_entities(text: str, method: str = "spacy") -> Set[str]:
    """
    Extract named entities from text using the best available method.
    
    Args:
        text: Input text string
        method: Extraction method ("auto", "stanza", "spacy", "rules")
        
    Returns:
        entities: Set of extracted entity strings
    """
    if method == "stanza":
        stanza_result = extract_entities_stanza(text)
        return stanza_result["entities"].union(stanza_result["mentions"]).union(stanza_result["concepts"])
    elif method == "spacy":
        return extract_entities_spacy(text)
    elif method == "rules":
        return extract_entities_rules(text)
    else:  # auto
        # Try Stanza first, then spaCy, then rules
        try:
            stanza_result = extract_entities_stanza(text)
            if stanza_result["entities"] or stanza_result["mentions"]:
                return stanza_result["entities"].union(stanza_result["mentions"]).union(stanza_result["concepts"])
        except Exception:
            pass
        
        try:
            return extract_entities_spacy(text)
        except Exception:
            return extract_entities_rules(text)


def extract_key_phrases(text: str) -> Set[str]:
    """
    Extract key phrases and compound terms that might not be caught by NER.
    
    Args:
        text: Input text string
        
    Returns:
        phrases: Set of key phrases
    """
    phrases = set()
    
    # Pattern for compound terms with specific indicators
    compound_patterns = [
        r'\b\w+(?:\s+\w+)*\s+(?:University|College|School|Institute|Foundation|Corporation|Company|Group)\b',
        r'\b(?:University|College|School|Institute|Foundation|Corporation|Company|Group)\s+of\s+\w+(?:\s+\w+)*\b',
        r'\b\w+(?:\s+\w+)*\s+(?:Act|Law|Treaty|Agreement|Convention)\b',
        r'\b(?:Mount|Lake|River|Valley|Bay|Ocean|Sea|Mountain)\s+\w+(?:\s+\w+)*\b',
        r'\b\w+(?:\s+\w+)*\s+(?:Stadium|Arena|Center|Centre|Building|Tower|Bridge)\b',
        r'\b(?:North|South|East|West|Northern|Southern|Eastern|Western)\s+\w+(?:\s+\w+)*\b',
    ]
    
    for pattern in compound_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        phrases.update([match.lower().strip() for match in matches])
    
    return phrases


def extract_coreference_chains(text: str) -> Dict[str, Set[str]]:
    """
    Extract coreference chains using Stanza (linking pronouns to entities).
    
    Args:
        text: Input text string
        
    Returns:
        chains: Dictionary mapping representative entities to their mentions
    """
    nlp = load_stanza_model()
    
    if nlp is None:
        return {}
    
    try:
        doc = nlp(text)
        chains = {}
        
        if hasattr(doc, 'coref_chains') and doc.coref_chains:
            for i, chain in enumerate(doc.coref_chains):
                # Find the most complete mention as representative
                representative = None
                max_length = 0
                mentions = set()
                
                for mention in chain.mentions:
                    mention_text = mention.text.strip()
                    mentions.add(mention_text.lower())
                    
                    if len(mention_text) > max_length:
                        max_length = len(mention_text)
                        representative = mention_text.lower()
                
                if representative:
                    chains[representative] = mentions
        
        return chains
    except Exception:
        return {}


def compute_entity_overlap(text1: str, text2: str, method: str = "spacy") -> float:
    """
    Compute entity overlap score between two texts using advanced NER.
    
    Args:
        text1: First text string
        text2: Second text string
        method: Entity extraction method
        
    Returns:
        overlap_score: Jaccard similarity of entities (0-1)
    """
    entities1 = extract_entities(text1, method)
    entities2 = extract_entities(text2, method)
    
    # Also include key phrases
    entities1.update(extract_key_phrases(text1))
    entities2.update(extract_key_phrases(text2))
    
    # For Stanza method, also consider coreference resolution
    if method in ["stanza", "auto"]:
        try:
            chains1 = extract_coreference_chains(text1)
            chains2 = extract_coreference_chains(text2)
            
            # Add all mentions from coreference chains
            for mentions in chains1.values():
                entities1.update(mentions)
            for mentions in chains2.values():
                entities2.update(mentions)
        except Exception:
            pass
    
    if not entities1 and not entities2:
        return 0.0
    
    intersection = len(entities1.intersection(entities2))
    union = len(entities1.union(entities2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_token_overlap(text1: str, text2: str, min_length: int = 3) -> float:
    """
    Compute token overlap score between two texts with improved preprocessing.
    
    Args:
        text1: First text string
        text2: Second text string
        min_length: Minimum token length to consider
        
    Returns:
        overlap_score: Jaccard similarity of tokens (0-1)
    """
    # Improved tokenization with better preprocessing
    def preprocess_and_tokenize(text: str) -> Set[str]:
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Extract tokens
        tokens = set(re.findall(r'\b\w+\b', text))
        
        # Filter out stopwords and short tokens
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'of', 'as',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        return {token for token in tokens 
                if len(token) >= min_length and token not in stopwords}
    
    tokens1 = preprocess_and_tokenize(text1)
    tokens2 = preprocess_and_tokenize(text2)
    
    if not tokens1 and not tokens2:
        return 0.0
    
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_overlap_features(question: str, paragraphs: List[str], 
                           entity_method: str = "rules") -> Dict[str, List[float]]:
    """
    Fallback function for single-example overlap computation.
    For batch processing, use batch_compute_overlap_features().
    
    Args:
        question: Question text
        paragraphs: List of paragraph texts
        entity_method: Method for entity extraction
        
    Returns:
        features: Dictionary with overlap feature arrays
    """
    # Use batch function for single example
    batch_result = batch_compute_overlap_features([question], [paragraphs], entity_method)
    return batch_result[0] if batch_result else {
        'qp_entity_overlap': [0.0] * len(paragraphs),
        'qp_token_overlap': [0.0] * len(paragraphs),
        'pp_entity_overlap': [[0.0] * len(paragraphs) for _ in range(len(paragraphs))],
        'pp_token_overlap': [[0.0] * len(paragraphs) for _ in range(len(paragraphs))]
    }


def batch_compute_overlap_features(questions: List[str], 
                                 paragraphs_list: List[List[str]], 
                                 entity_method: str = "rules") -> List[Dict[str, List[float]]]:
    """
    OPTIMIZED: Batch compute overlap features for many question-paragraph sets.
    
    Args:
        questions: List of question texts
        paragraphs_list: List of paragraph lists (one per question)
        entity_method: Method for entity extraction
        
    Returns:
        batch_features: List of overlap feature dictionaries
    """
    from tqdm import tqdm
    import time
    
    print(f"   🔧 Batch processing {len(questions)} questions with {entity_method} method...")
    start_time = time.time()
    
    # OPTIMIZATION 1: Extract entities for all texts at once
    all_texts = []
    text_indices = {}  # Map text -> index in all_texts
    
    # Collect all unique texts
    for i, (question, paragraphs) in enumerate(zip(questions, paragraphs_list)):
        if question not in text_indices:
            text_indices[question] = len(all_texts)
            all_texts.append(question)
        
        for para in paragraphs:
            if para not in text_indices:
                text_indices[para] = len(all_texts)
                all_texts.append(para)
    
    print(f"   📝 Extracting entities from {len(all_texts)} unique texts...")
    
    # OPTIMIZATION 2: Batch extract entities (much faster)
    if entity_method == "rules":
        # Use fast rule-based extraction
        all_entities = [extract_entities_rules(text) for text in tqdm(all_texts, desc="Entity extraction", leave=False)]
    else:
        # For spacy/stanza, still extract individually but with progress
        all_entities = [extract_entities(text, entity_method) for text in tqdm(all_texts, desc="Entity extraction", leave=False)]
    
    entity_time = time.time() - start_time
    print(f"   ✅ Entity extraction completed in {entity_time:.1f}s")
    
    # OPTIMIZATION 3: Fast overlap computation using cached entities
    print(f"   🧮 Computing overlaps...")
    start_time = time.time()
    
    batch_features = []
    
    for question, paragraphs in tqdm(zip(questions, paragraphs_list), desc="Computing overlaps", leave=False):
        n = len(paragraphs)
        
        # Get cached entities
        q_entities = all_entities[text_indices[question]]
        p_entities = [all_entities[text_indices[para]] for para in paragraphs]
        
        # Fast overlap computation
        qp_entity_overlap = []
        for p_ent in p_entities:
            if not q_entities and not p_ent:
                overlap = 0.0
            elif not q_entities or not p_ent:
                overlap = 0.0
            else:
                intersection = len(q_entities.intersection(p_ent))
                union = len(q_entities.union(p_ent))
                overlap = intersection / union if union > 0 else 0.0
            qp_entity_overlap.append(overlap)
        
        # Paragraph-paragraph overlaps
        pp_entity_overlap = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    pp_entity_overlap[i][j] = 1.0  # Self-overlap
                else:
                    p1_ent = p_entities[i]
                    p2_ent = p_entities[j]
                    
                    if not p1_ent and not p2_ent:
                        overlap = 0.0
                    elif not p1_ent or not p2_ent:
                        overlap = 0.0
                    else:
                        intersection = len(p1_ent.intersection(p2_ent))
                        union = len(p1_ent.union(p2_ent))
                        overlap = intersection / union if union > 0 else 0.0
                    
                    pp_entity_overlap[i][j] = overlap
        
        # Store results (simplified - just entity overlap for speed)
        batch_features.append({
            'qp_entity_overlap': qp_entity_overlap,
            'qp_token_overlap': [0.0] * n,  # Skip token overlap for speed
            'pp_entity_overlap': pp_entity_overlap,
            'pp_token_overlap': [[0.0] * n for _ in range(n)]  # Skip for speed
        })
    
    overlap_time = time.time() - start_time
    total_time = entity_time + overlap_time
    
    print(f"   ✅ Overlap computation completed in {overlap_time:.1f}s")
    print(f"   🎯 Total batch processing: {total_time:.1f}s ({total_time/len(questions)*1000:.1f}ms per question)")
    
    return batch_features


def analyze_entities(text: str) -> Dict[str, Any]:
    """
    Analyze entities in text for debugging and inspection.
    
    Args:
        text: Input text
        
    Returns:
        analysis: Dictionary with entity analysis
    """
    # Get results from all methods
    stanza_result = extract_entities_stanza(text) if load_stanza_model() else {"entities": set(), "mentions": set(), "concepts": set()}
    spacy_entities = extract_entities_spacy(text) if load_spacy_model() else set()
    rule_entities = extract_entities_rules(text)
    key_phrases = extract_key_phrases(text)
    coref_chains = extract_coreference_chains(text)
    
    # Combine all Stanza results
    stanza_all = stanza_result["entities"].union(stanza_result["mentions"]).union(stanza_result["concepts"])
    
    return {
        'stanza_entities': sorted(stanza_result["entities"]),
        'stanza_mentions': sorted(stanza_result["mentions"]),
        'stanza_concepts': sorted(stanza_result["concepts"]),
        'stanza_all': sorted(stanza_all),
        'spacy_entities': sorted(spacy_entities),
        'rule_entities': sorted(rule_entities),
        'key_phrases': sorted(key_phrases),
        'coref_chains': {k: sorted(v) for k, v in coref_chains.items()},
        'combined': sorted(stanza_all.union(spacy_entities).union(rule_entities).union(key_phrases)),
        'stanza_only': sorted(stanza_all - spacy_entities - rule_entities),
        'spacy_only': sorted(spacy_entities - stanza_all - rule_entities),
        'rules_only': sorted(rule_entities - stanza_all - spacy_entities)
    }


def get_entity_relations(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract entity relations using dependency parsing (Stanza).
    
    Args:
        text: Input text
        
    Returns:
        relations: List of (subject, relation, object) tuples
    """
    nlp = load_stanza_model()
    
    if nlp is None:
        return []
    
    try:
        doc = nlp(text)
        relations = []
        
        for sentence in doc.sentences:
            # Simple relation extraction based on dependency parsing
            for word in sentence.words:
                if word.deprel in ['nsubj', 'nsubj:pass'] and word.head > 0:
                    head_word = sentence.words[word.head - 1]
                    
                    # Look for objects
                    for other_word in sentence.words:
                        if (other_word.head == word.head and 
                            other_word.deprel in ['dobj', 'iobj', 'nmod', 'obl']):
                            
                            relations.append((
                                word.text.lower(),
                                head_word.text.lower(),
                                other_word.text.lower()
                            ))
        
        return relations
    except Exception:
        return [] 