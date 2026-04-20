# ============================================================
# your_pipeline.py (burak_pipeline.py)
# Auto-extracted from burak_connections_pipeline.ipynb
# Entry point: run_full_pipeline()
# ============================================================

import os
import re
import json
import random
import requests
from pathlib import Path
import anthropic
import openai
import nltk
import gensim.downloader as api

from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet, words as nltk_words, cmudict
from Levenshtein import distance as lev

try:
    from IPython.display import HTML, display
except ImportError:
    def display(x):  # noqa: ANN001
        print(x)

    def HTML(_x):  # noqa: ANN001
        return None


nltk.download('words', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('cmudict', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

try:
    from google.colab import userdata

    ANTHROPIC_KEY = userdata.get('ANTHROPIC_API_KEY')
    OPENAI_KEY = userdata.get('OPENAI_API_KEY')
except Exception:
    ANTHROPIC_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
    OPENAI_KEY = os.environ.get('OPENAI_API_KEY', '')

claude_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_KEY)

# --- Cell 2 ---
# Word lists
try:
    resp = requests.get('https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt', timeout=10)
    COMMON_WORDS = set(w.strip().lower() for w in resp.text.splitlines() if w.strip().isalpha() and len(w.strip()) >= 3)
    print(f'Common words : {len(COMMON_WORDS):,}')
except Exception:
    COMMON_WORDS = set(w.lower() for w in nltk_words.words() if w.isalpha() and 3 <= len(w) <= 10)

FULL_DICT   = set(w.lower() for w in nltk_words.words() if w.isalpha())
SHORT_WORDS = set(w for w in COMMON_WORDS if 3 <= len(w) <= 6)
CMU         = cmudict.dict()
print(f'Full dict    : {len(FULL_DICT):,}  Short: {len(SHORT_WORDS):,}  CMU: {len(CMU):,}')

_WV_MODEL = None


def _get_wv():
    """
    Lazy-load word2vec Google News 300 (~1.6GB on first download; Gensim caches
    under GENSIM_DATA_DIR or ~/gensim-data). No RAM cost until a pipeline step
    needs embeddings.
    """
    global _WV_MODEL
    if _WV_MODEL is None:
        print('Loading word2vec-google-news-300 (first run downloads ~1.6GB; cached on disk after that)...')
        _WV_MODEL = api.load('word2vec-google-news-300')
        print('word2vec loaded.')
    return _WV_MODEL


# --- Cell 3 ---
# Proper noun filter
PROPER_NOUNS_NLTK = set(w.lower() for w in nltk_words.words() if w[0].isupper() and w.lower() not in COMMON_WORDS)
PROPER_NOUN_BLOCKLIST = {
    'carmen','paris','jordan','china','iris','mark','april','may','june','august',
    'virginia','georgia','florence','amber','sandy','dusty','sunny','holly','ivy',
    'cliff','wade','chase','hunter','faith','grace','hope','joy','lance','rod',
    'dean','glen','dale','lee','gene','don','rex','bart','clay','cole','dawn',
    'earl','finn','ford','gil','han','jack','jade','jan','jay','jean','jen',
    'jim','joe','jon','josh','juan','kai','kent','kim','kirk','kurt','lane',
    'lars','leon','liz','lou','luke','luna','lynn','marc','mario','mars','matt',
    'max','miles','milo','ming','morgan','neil','nick','noel','norm','otto','owen',
    'pat','paul','pete','phil','ray','reed','rob','ron','ross','roy','russ',
    'ruth','ryan','sam','sara','saul','sean','seth','skip','stan','sue','ted',
    'tim','todd','tom','troy','val','vera','walt','ward','wayne','webb','will','wren',
}

def is_proper_noun(w): return w.lower() in PROPER_NOUN_BLOCKLIST or w.lower() in PROPER_NOUNS_NLTK
def group_has_proper_nouns(g): return any(is_proper_noun(w) for w in g)

# Shared validity helpers
def get_lemmas(word):
    w = word.lower()
    s = {w}
    for p in ['n','v','a','r','s']: s.add(lemmatizer.lemmatize(w, pos=p))
    return s

def word_root(word):
    w = word.lower()
    return stemmer.stem(min([w, lemmatizer.lemmatize(w,'n'), lemmatizer.lemmatize(w,'v')], key=len))

def has_shared_root(words):
    roots = [word_root(w) for w in words]
    return len(set(roots)) < len(roots)

def has_too_similar(words, max_d=2):
    for a,b in combinations(words, 2):
        if lev(a.lower(),b.lower()) <= max_d or a.lower() in b.lower() or b.lower() in a.lower():
            return True
    return False

def group_has_conjugation_issues(group):
    words = [w.lower() for w in group]
    for i,w1 in enumerate(words):
        for w2 in words[i+1:]:
            if w1 in w2 or w2 in w1: return True
    for i,w1 in enumerate(words):
        for w2 in words[i+1:]:
            if get_lemmas(w1) & get_lemmas(w2): return True
    return False

def is_valid_group(words):
    if len(words)!=4 or len(set(w.lower() for w in words))!=4: return False
    if has_shared_root(words): return False
    if has_too_similar(words): return False
    if group_has_proper_nouns(words): return False
    return True

def extract_first_json(raw):
    raw = re.sub(r'```(?:json)?','',raw).strip()
    s = raw.find('{')
    if s==-1: raise ValueError(f'No JSON: {raw}')
    depth,end = 0,-1
    for i,ch in enumerate(raw[s:],s):
        if ch=='{': depth+=1
        elif ch=='}': depth-=1
        if depth==0: end=i; break
    return json.loads(raw[s:end+1])

print('Utilities loaded.')

# --- Cell 4 ---
# Load teammate files (all optional). Paths are next to this file so CWD does not matter.
_BURAK_DIR = Path(__file__).resolve().parent

import joblib, pandas as pd

CLASSIFIER = None
_classifier_path = _BURAK_DIR / 'purple_group_classifier.joblib'
try:
    CLASSIFIER = joblib.load(os.fspath(_classifier_path))
    print('Classifier loaded.')
except FileNotFoundError:
    print(
        'No classifier — place purple_group_classifier.joblib next to burak_pipeline.py '
        '(purple ML scores default to 0.5 for every candidate until then).'
    )
except Exception as e:
    print(f'Classifier failed to load ({type(e).__name__}: {e}); purple scores default to 0.5.')

SCORE_WV = None
_vectors_path = _BURAK_DIR / 'vectors.bin'
try:
    from gensim.models import KeyedVectors
    SCORE_WV = KeyedVectors.load_word2vec_format(
        os.fspath(_vectors_path), binary=True, unicode_errors='ignore'
    )
    print('vectors.bin loaded.')
except FileNotFoundError:
    print('No vectors.bin — using full word2vec for classifier features (slower RAM).')
except Exception as e:
    print(f'vectors.bin load failed ({e}); using full word2vec for scoring.')

COMPOUNDS_BY_PREFIX, COMPOUNDS_BY_SUFFIX = defaultdict(list), defaultdict(list)
try:
    with open(os.fspath(_BURAK_DIR / 'compounds_natural.txt')) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts)==3:
                l,r,_ = parts
                COMPOUNDS_BY_PREFIX[l].append(r)
                COMPOUNDS_BY_SUFFIX[r].append(l)
    print(f'Compounds: {len(COMPOUNDS_BY_PREFIX)} prefixes, {len(COMPOUNDS_BY_SUFFIX)} suffixes.')
except: print('No compounds file.')

ANAGRAM_BUCKETS = defaultdict(list)
try:
    with open(os.fspath(_BURAK_DIR / 'anagram_words.txt')) as f:
        for line in f:
            w = line.strip()
            if w: ANAGRAM_BUCKETS[''.join(sorted(w.lower()))].append(w.lower())
    print(f'Anagram buckets: {sum(1 for v in ANAGRAM_BUCKETS.values() if len(set(v))>=4)} with 4+ words.')
except: print('No anagram file.')

VERB_NOUN_GROUPS = []
try:
    with open(os.fspath(_BURAK_DIR / 'verb_noun_associations_llm_keep2.txt')) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts)==3:
                verb = parts[0]
                nouns = [n.strip() for n in parts[2].split(',') if n.strip()]
                if len(nouns)>=4: VERB_NOUN_GROUPS.append((verb, nouns))
    print(f'Verb-noun: {len(VERB_NOUN_GROUPS)} groups.')
except: print('No verb_noun file.')

# --- Cell 5 ---
# === ALL 8 PURPLE GENERATORS ===

HIDDEN_WORD_THEMES = {
    'animal': ['ant','ape','bat','bear','bee','boar','bull','cat','cod','cow','cub','doe','dog','eel','elk','ewe','fly','fox','gnu','hen','hog','jay','koi','lion','lynx','mare','mink','mole','moth','mule','newt','owl','pig','pug','ram','rat','ray','seal','slug','stag','swan','toad','vole','wasp','wolf','worm','wren','yak','asp','carp','clam','colt','crab','crow','deer','dove','duck','fawn','flea','foal','frog','gnat','goat','grub','gull','hare','hawk','ibis','kite','lamb','lark','myna','pony','puma','rook','snail','sole','teal','tern','tick','trout','viper'],
    'color': ['red','tan','pink','jade','gold','rose','navy','plum','rust','sage','sand','teal','amber','azure','beige','ebony','ivory','khaki','lilac','mauve','ochre','olive','pearl','sepia','taupe','umber'],
    'number': ['one','two','ten','six','eight','nine','zero','four','five','three'],
    'body part': ['ear','arm','rib','hip','lip','shin','toe','knee','eye','jaw','leg','lap','neck','chin','heel','palm','nail','vein','loin','brow','calf','lash','pore','gum'],
}
HIDDEN_CANDIDATES = [(w,t) for t,ws in HIDDEN_WORD_THEMES.items() for w in ws]

PREFIX_CONNECTORS = ['under','over','out','back','down','up','counter','cross','fore','hand','head','high','home','hot','land','life','light','long','low','mid','night','road','sea','side','sky','step','sun','time','water','well','wild','wind','wood','work']
SUFFIX_CONNECTORS = ['house','work','man','out','side','line','light','land','time','way','place','back','yard','down','hand','head','life','long','mark','book','word','play','ball','board','break','burn','call','cap','card','case','cut','day','drop','fall','field','fire','fish','floor','fly','foot','gate','ground','gun','hill','hold','hole','hook','horn','keep','key','lock','look','lot','master','mate','mill','mouth','note','pack','page','park','pass','path','pay','pick','pipe','point','pool','port','post','pot','power','print','proof','rack','rain','rise','road','rock','roll','roof','room','root','rope','round','run','rush','sale','sand','sea','seat','set','shed','shell','ship','shop','shot','show','sight','sign','skin','smoke','snow','song','stone','stop','storm','stream','street','suit','swing','tail','take','talk','tide','tip','tone','tool','top','touch','town','track','trade','trail','train','tree','turn','view','walk','wall','ward','watch','water','wave','wheel','wind','wing','wire','wood']
COMMON_SUFFIXES = {'ing','er','ed','est','ly','ness','tion','ion','s','es','ful','less','able','ible','ment','ity','ize','ise','ify','al','ous','ive','ary','ery'}

def is_trivial(prefix, word): return word[len(prefix):].lower() in COMMON_SUFFIXES

def gen_hidden_word():
    cands = HIDDEN_CANDIDATES.copy(); random.shuffle(cands)
    for hidden,theme in cands:
        pool = [w for w in COMMON_WORDS if len(w)>=6 and w!=hidden and w in FULL_DICT and 0<w.find(hidden)<len(w)-len(hidden)]
        if len(pool)<6: continue
        random.shuffle(pool)
        for s in range(len(pool)-3):
            c = pool[s:s+4]
            if is_valid_group(c):
                return {'mechanism':'hidden_word','answer':hidden.upper(),'group':[w.upper() for w in c],'connection':f'Each contains a hidden {theme} ("{hidden.upper()}")','examples':[w.upper().replace(hidden.upper(),f'[{hidden.upper()}]') for w in c]}
    return None

def gen_prefix():
    cs = PREFIX_CONNECTORS.copy(); random.shuffle(cs)
    for c in cs:
        pool = [w for w in COMMON_WORDS if len(w)>=3 and w!=c and w in FULL_DICT and (c+w) in FULL_DICT and (c+w) in COMMON_WORDS]
        if len(pool)<5: continue
        random.shuffle(pool)
        for s in range(len(pool)-3):
            chosen = pool[s:s+4]
            if is_valid_group(chosen):
                return {'mechanism':'prefix','answer':c.upper(),'group':[w.upper() for w in chosen],'connection':f'"{c.upper()}" + ___','examples':[f'{c.upper()}+{w.upper()}={(c+w).upper()}' for w in chosen]}
    return None

def gen_suffix():
    cs = SUFFIX_CONNECTORS.copy(); random.shuffle(cs)
    for c in cs:
        pool = [w for w in COMMON_WORDS if len(w)>=3 and w!=c and w in FULL_DICT and (w+c) in FULL_DICT and (w+c) in COMMON_WORDS]
        if len(pool)<5: continue
        random.shuffle(pool)
        for s in range(len(pool)-3):
            chosen = pool[s:s+4]
            if is_valid_group(chosen):
                return {'mechanism':'suffix','answer':c.upper(),'group':[w.upper() for w in chosen],'connection':f'___ + "{c.upper()}"','examples':[f'{w.upper()}+{c.upper()}={(w+c).upper()}' for w in chosen]}
    return None

def gen_synonym_prefix():
    seeds = [w for w in COMMON_WORDS if 4<=len(w)<=8]; random.shuffle(seeds)
    for seed in seeds[:300]:
        syns = set()
        for ss in wordnet.synsets(seed)[:3]:
            for l in ss.lemmas():
                s = l.name().lower().replace('_','')
                if s.isalpha() and 2<=len(s)<=6 and s!=seed and s in COMMON_WORDS: syns.add(s)
        if len(syns)<4: continue
        matches, used = [], []
        for syn in syns:
            cands = [w for w in COMMON_WORDS if w.startswith(syn) and w!=syn and len(w)>=max(6,len(syn)+3) and w in FULL_DICT and w not in used and w!=seed and not is_trivial(syn,w) and not is_proper_noun(w)]
            if cands:
                cands.sort(key=lambda w: abs(len(w)-(len(syn)+5)))
                matches.append((syn,cands[0])); used.append(cands[0])
        if len(matches)<4: continue
        outers = [m[1] for m in matches[:4]]
        if is_valid_group(outers):
            return {'mechanism':'synonym_prefix','answer':seed.upper(),'group':[w.upper() for w in outers],'connection':f'Starting with synonyms for "{seed.upper()}"','examples':[f'{s.upper()}→{o.upper()}' for s,o in matches[:4]]}
    return None

def gen_compound_prefix():
    if not COMPOUNDS_BY_PREFIX: return None
    keys = list(COMPOUNDS_BY_PREFIX.keys()); random.shuffle(keys)
    for k in keys:
        bases = list(set(COMPOUNDS_BY_PREFIX[k])); random.shuffle(bases)
        for combo in combinations(bases[:20],4):
            if is_valid_group(list(combo)):
                return {'mechanism':'compound_prefix','answer':k.upper(),'group':[w.upper() for w in combo],'connection':f'"{k.upper()}" + ___','examples':[f'{k.upper()}+{w.upper()}' for w in combo]}
    return None

def gen_compound_suffix():
    if not COMPOUNDS_BY_SUFFIX: return None
    keys = list(COMPOUNDS_BY_SUFFIX.keys()); random.shuffle(keys)
    for k in keys:
        bases = list(set(COMPOUNDS_BY_SUFFIX[k])); random.shuffle(bases)
        for combo in combinations(bases[:20],4):
            if is_valid_group(list(combo)):
                return {'mechanism':'compound_suffix','answer':k.upper(),'group':[w.upper() for w in combo],'connection':f'___ + "{k.upper()}"','examples':[f'{w.upper()}+{k.upper()}' for w in combo]}
    return None

def gen_anagram():
    valid = [(k,list(set(v))) for k,v in ANAGRAM_BUCKETS.items() if len(set(v))>=4]
    if not valid: return None
    random.shuffle(valid)
    for k,words in valid:
        random.shuffle(words)
        for combo in combinations(words,4):
            if is_valid_group(list(combo)):
                return {'mechanism':'anagram','answer':'anagram','group':[w.upper() for w in combo],'connection':'Anagrams of each other','examples':[w.upper() for w in combo]}
    return None

def gen_verb_noun():
    if not VERB_NOUN_GROUPS: return None
    groups = VERB_NOUN_GROUPS.copy(); random.shuffle(groups)
    for verb,nouns in groups:
        if is_valid_group(nouns[:4]):
            return {'mechanism':'verb_noun','answer':verb.upper(),'group':[w.upper() for w in nouns[:4]],'connection':f'Things you can {verb.upper()}','examples':[w.upper() for w in nouns[:4]]}
    return None

print('All 8 generators ready.')

# --- Cell 6 ---
# ============================================================
# CELL 3c — ML Scoring + Master Purple Generator
# ============================================================

def score_candidate(c):
    if CLASSIFIER is None: return 0.5
    model = SCORE_WV if SCORE_WV else _get_wv()
    try:
        mech   = c['mechanism']
        answer = c['answer'].lower().replace(' ', '')
        words  = [w.lower() for w in c['group']]
        miss   = len(model.key_to_index) + 1
        ranks  = [model.key_to_index.get(w, miss) for w in words]
        ar     = model.key_to_index.get(answer, miss)
        mmap   = {
            'prefix':         'compound_prefix',
            'suffix':         'compound_suffix',
            'hidden_word':    'compound_prefix',
            'synonym_prefix': 'compound_prefix',
            'verb_noun':      'verb_noun_association',
        }
        cm      = mmap.get(mech, mech)
        cl      = (
            [len(answer + w) for w in words] if cm == 'compound_prefix'
            else [len(w + answer) for w in words] if cm == 'compound_suffix'
            else [0, 0, 0, 0]
        )
        all_lev = [lev(a, b) for a, b in combinations(words, 2)]

        features = pd.DataFrame([{
            'mechanism':                        cm,
            'answer_length':                    len(answer),
            'min_word_length':                  min(len(w) for w in words),
            'max_word_length':                  max(len(w) for w in words),
            'avg_word_length':                  sum(len(w) for w in words) / 4,
            'max_length_gap':                   max(len(w) for w in words) - min(len(w) for w in words),
            'min_pairwise_levenshtein':         min(all_lev),
            'answer_overlaps_words':            int(any(answer in w or w in answer for w in words)),
            'words_share_root':                 int(has_shared_root(words)),
            'has_very_similar_spelling':        int(min(all_lev) <= 2),
            'all_words_same_anagram_signature': int(len(set(''.join(sorted(w)) for w in words)) == 1),
            'avg_compound_length':              sum(cl) / 4,
            'max_compound_length':              max(cl),
            'min_wv_rank':                      min(ranks),
            'max_wv_rank':                      max(ranks),
            'avg_wv_rank':                      sum(ranks) / 4,
            'answer_wv_rank':                   ar,
            'unknown_wv_count':                 sum(1 for r in ranks if r == miss),
        }])

        return float(CLASSIFIER.predict_proba(features)[:, 1][0])

    except Exception as e:
        return 0.5


def generate_purple_group(n_candidates=8):
    """
    Try all available mechanisms, collect up to n_candidates,
    score each with the ML classifier, return the best.
    """
    generators = [
        ('hidden_word',     gen_hidden_word),
        ('prefix',          gen_prefix),
        ('suffix',          gen_suffix),
        ('synonym_prefix',  gen_synonym_prefix),
        ('compound_prefix', gen_compound_prefix),
        ('compound_suffix', gen_compound_suffix),
        ('anagram',         gen_anagram),
        ('verb_noun',       gen_verb_noun),
    ]
    random.shuffle(generators)

    candidates = []
    for name, fn in generators:
        r = fn()
        if r is None:
            print(f'  [{name}] no candidate')
            continue
        score = score_candidate(r)
        tag   = 'keep' if score >= 0.8 else ('borderline' if score >= 0.3 else 'reject')
        print(f'  [{name}] score={score:.3f} ({tag}) {r["group"]}')
        candidates.append((score, r))
        if len(candidates) >= n_candidates:
            break

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[0])
    best_score, best = candidates[0]
    print(f'\n  Best: [{best["mechanism"]}] score={best_score:.3f}')
    return best

print('Scoring + master generator ready.')

# --- Cell 7 ---
# ── Run purple + select TWO impostors ────────────────────────

def find_impostor(purple_word: str,
                   purple_group: list,
                   exclude_extra: list = [],
                   topn: int = 200) -> str | None:
    anchor_lower = purple_word.lower()
    excluded     = set(w.lower() for w in purple_group + exclude_extra)

    key = next(
        (k for k in [anchor_lower, purple_word.upper(), purple_word.capitalize()] if k in _get_wv()),
        None
    )
    if key is None:
        return None

    try:
        anchor_pos = pos_tag([anchor_lower])[0][1]
    except Exception:
        anchor_pos = None

    for word, _ in _get_wv().most_similar(key, topn=topn):
        word = word.lower().strip()
        if '_' in word:
            continue
        if word not in COMMON_WORDS or word not in FULL_DICT:
            continue
        if word in excluded:
            continue
        if lev(anchor_lower, word) / max(len(anchor_lower), len(word)) < 0.3:
            continue
        if word in anchor_lower or anchor_lower in word:
            continue
        if get_lemmas(word) & get_lemmas(anchor_lower):
            continue
        if any(get_lemmas(word) & get_lemmas(p.lower()) for p in purple_group):
            continue
        if anchor_pos:
            try:
                word_pos = pos_tag([word])[0][1]
                if word_pos[0] != anchor_pos[0]:
                    continue
            except Exception:
                pass
        return word

    return None


def find_impostor_rhymes(purple_word: str,
                          purple_group: list,
                          exclude_extra: list = [],
                          min_rhyme_pool: int = 4) -> str | None:
    """
    Find an impostor that:
      - Is semantically similar to purple_word (word2vec)
      - Has at least min_rhyme_pool other common words that rhyme with it
        (so it can anchor a full rhyme group)
    """
    anchor_lower = purple_word.lower()
    excluded     = set(w.lower() for w in purple_group + exclude_extra)

    key = next(
        (k for k in [anchor_lower, purple_word.upper(), purple_word.capitalize()] if k in _get_wv()),
        None
    )
    if key is None:
        return None

    for word, _ in _get_wv().most_similar(key, topn=300):
        word = word.lower().strip()
        if '_' in word:
            continue
        if word not in COMMON_WORDS or word not in FULL_DICT:
            continue
        if word in excluded:
            continue
        if lev(anchor_lower, word) / max(len(anchor_lower), len(word)) < 0.3:
            continue
        if word in anchor_lower or anchor_lower in word:
            continue
        if get_lemmas(word) & get_lemmas(anchor_lower):
            continue
        if any(get_lemmas(word) & get_lemmas(p.lower()) for p in purple_group):
            continue

        # Check it can anchor a rhyme group
        ending = get_rhyme_ending(word)
        if ending is None:
            continue
        rhyme_pool = [
            w for w in rhyme_idx.get(ending, [])
            if w != word
            and w not in excluded
            and len(w) >= 3
            and not is_proper_noun(w)
        ]
        if len(rhyme_pool) >= min_rhyme_pool:
            return word

    return None


def find_impostor_w2v_seed(purple_word: str,
                             purple_group: list,
                             exclude_extra: list = [],
                             min_similar: int = 3) -> str | None:
    """
    Find an impostor that:
      - Is semantically similar to purple_word (word2vec)
      - Has at least min_similar common words clustering around it
        (so it can anchor a yellow group)
    """
    anchor_lower = purple_word.lower()
    excluded     = set(w.lower() for w in purple_group + exclude_extra)

    key = next(
        (k for k in [anchor_lower, purple_word.upper(), purple_word.capitalize()] if k in _get_wv()),
        None
    )
    if key is None:
        return None

    for word, _ in _get_wv().most_similar(key, topn=300):
        word = word.lower().strip()
        if '_' in word:
            continue
        if word not in COMMON_WORDS or word not in FULL_DICT:
            continue
        if word in excluded:
            continue
        if lev(anchor_lower, word) / max(len(anchor_lower), len(word)) < 0.3:
            continue
        if word in anchor_lower or anchor_lower in word:
            continue
        if get_lemmas(word) & get_lemmas(anchor_lower):
            continue
        if any(get_lemmas(word) & get_lemmas(p.lower()) for p in purple_group):
            continue

        # Check it can seed a yellow group
        sim = find_similar_w2v(word, list(excluded) + [word], n=3)
        if len(sim) >= min_similar:
            return word

    return None


def label_fake_connection(anchor: str, imp1: str, imp2: str) -> str:
    prompt = (
        f'You are designing a NYT Connections puzzle red herring.\n\n'
        f'Three words: {anchor.upper()}, {imp1.upper()}, {imp2.upper()}\n\n'
        f'Write a short NYT Connections-style label for what these three words '
        f'SEEM to have in common. Be specific and convincing.\n\n'
        f'Good examples:\n'
        f'- "Things you can do to a ball"\n'
        f'- "Words that follow \'cold\'"\n'
        f'- "Synonyms for quit"\n\n'
        f'Bad examples (too vague):\n'
        f'- "Things that sound like verbs"\n'
        f'- "Words related to work"\n\n'
        f'The label must genuinely apply to all three words.\n'
        f'Max 6 words. Return only the label, nothing else.'
    )
    try:
        resp  = claude_client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=20,
            messages=[{'role': 'user', 'content': prompt}]
        )
        label = resp.content[0].text.strip().strip('"')

        # Retry with Sonnet if label is too generic
        generic_flags = ['verb', 'noun', 'sound', 'concept', 'related', 'word']
        if len(label) < 8 or any(f in label.lower() for f in generic_flags):
            resp2 = claude_client.messages.create(
                model='claude-sonnet-4-6',
                max_tokens=20,
                messages=[{'role': 'user', 'content': prompt}]
            )
            label = resp2.content[0].text.strip().strip('"')

        return label
    except Exception:
        return f'Things associated with {anchor.upper()}'


def select_two_impostors(purple_group: list) -> dict | None:
    """
    imp1 → green (must be able to anchor a rhyme group)
    imp2 → yellow (must be able to seed a word2vec group)

    Tries each purple word as anchor in random order.
    """
    candidates = purple_group.copy()
    random.shuffle(candidates)

    for purple_word in candidates:
        print(f'  Trying anchor: {purple_word}')

        # imp1 must be able to anchor a rhyme group
        imp1 = find_impostor_rhymes(
            purple_word=purple_word,
            purple_group=purple_group,
            exclude_extra=[],
        )
        if imp1 is None:
            print(f'    No rhyme-capable imp1 found')
            continue

        print(f'    imp1 (green anchor): {imp1.upper()}')

        # imp2 must be able to seed a yellow group
        imp2 = find_impostor_w2v_seed(
            purple_word=purple_word,
            purple_group=purple_group,
            exclude_extra=[imp1],
        )
        if imp2 is None:
            print(f'    No w2v-capable imp2 found')
            continue

        print(f'    imp2 (yellow seed): {imp2.upper()}')

        fake_label = label_fake_connection(purple_word, imp1, imp2)
        print(f'    Fake connection: "{fake_label}"')

        return {
            'anchor':          purple_word.upper(),
            'imp1':            imp1.upper(),   # → green (rhyme anchor)
            'imp2':            imp2.upper(),   # → yellow (w2v seed)
            'imp1_target':     'green',
            'imp2_target':     'yellow',
            'fake_connection': fake_label,
        }

    return None


# --- Cell 8 ---
# ============================================================
# CELL 4 — Green Group
#
# Mechanism probabilities:
#   50% — rhyme (CMU phoneme ending)
#   25% — anagram (all words are anagrams of each other)
#   25% — letter pattern (all words contain same cluster)
#
# If an impostor is assigned to green, it is used as the
# ANCHOR — the mechanism is chosen based on what the impostor
# word can support, then 3 companions are found to match it.
# ============================================================

# ── Rhyme utilities ───────────────────────────────────────────

def _ending_from_phones(phones):
    """Rhyme nucleus: last stressed vowel (CMU stress 1 or 2) and following phonemes."""
    last = None
    for i, p in enumerate(phones):
        if len(p) >= 1 and p[-1] in '12':
            last = i
    if last is None:
        return None
    return ' '.join(phones[last:])


def get_rhyme_endings(word):
    """All distinct rhyme endings (one per pronunciation) for word."""
    w = word.lower()
    if w not in CMU:
        return []
    seen, out = set(), []
    for phones in CMU[w]:
        e = _ending_from_phones(phones)
        if e and e not in seen:
            seen.add(e)
            out.append(e)
    return out


def get_rhyme_ending(word):
    ends = get_rhyme_endings(word)
    return ends[0] if ends else None


print('Building rhyme index...')
rhyme_idx = {}
for w in COMMON_WORDS:
    if w not in FULL_DICT or w not in CMU:
        continue
    for phones in CMU[w]:
        e = _ending_from_phones(phones)
        if e:
            rhyme_idx.setdefault(e, []).append(w)
for _e in rhyme_idx:
    rhyme_idx[_e] = list(dict.fromkeys(rhyme_idx[_e]))

VALID_ENDINGS = {e: ws for e, ws in rhyme_idx.items() if len(ws) >= 6}
print(f'Valid rhyme endings   : {len(VALID_ENDINGS):,}')


# ── Anagram utilities ─────────────────────────────────────────

def get_anagram_key(word):
    return ''.join(sorted(word.lower()))


print('Building anagram index...')
anagram_idx = {}
for w in COMMON_WORDS:
    if w not in FULL_DICT: continue
    key = get_anagram_key(w)
    anagram_idx.setdefault(key, []).append(w)

VALID_ANAGRAM_GROUPS = {
    k: list(set(v)) for k, v in anagram_idx.items()
    if len(set(v)) >= 4
}
print(f'Valid anagram groups  : {len(VALID_ANAGRAM_GROUPS)}')


# ── Letter pattern utilities ──────────────────────────────────

LETTER_PATTERNS = [
    'ght', 'tch', 'ck', 'wh', 'kn', 'wr', 'mb',
    'ph', 'qu', 'dge', 'nch', 'rth', 'lth',
    'scr', 'shr', 'spl', 'spr', 'str', 'thr',
    'tion', 'sion', 'ough', 'augh', 'eigh', 'igh',
]

print('Building letter pattern index...')
pattern_idx = {}
for pattern in LETTER_PATTERNS:
    pool = [
        w for w in COMMON_WORDS
        if pattern in w.lower()
        and w in FULL_DICT
        and len(w) >= 4
        and not is_proper_noun(w)
    ]
    if len(pool) >= 8:
        pattern_idx[pattern] = pool

print(f'Valid letter patterns : {len(pattern_idx)}')


# ── inject_impostor ───────────────────────────────────────────

def inject_impostor(group: list, impostor: str, mechanism: str = '') -> tuple:
    """
    Fallback only — used when anchor-based generation fails.
    For rhyme: replace word with most similar visual ending.
    For others: replace word most semantically similar via word2vec.
    Returns (new_group, replaced_word).
    """
    group          = group.copy()
    impostor_lower = impostor.lower()

    if mechanism == 'rhyme':
        imp_set = set(get_rhyme_endings(impostor_lower))
        word_ends = [get_rhyme_ending(w.lower()) for w in group]
        best_idx = None
        # Prefer swapping out a word whose CMU ending does not match the impostor's (bad slot).
        if imp_set:
            for i, we in enumerate(word_ends):
                if we and we not in imp_set:
                    best_idx = i
                    break
        # Else swap the odd-one-out vs the group's dominant ending.
        if best_idx is None:
            cnt = Counter(e for e in word_ends if e)
            ref = cnt.most_common(1)[0][0] if cnt else None
            if ref:
                for i, we in enumerate(word_ends):
                    if we != ref:
                        best_idx = i
                        break
        # Orthographic fallback (legacy behavior).
        if best_idx is None:
            imp_end = impostor_lower[-3:]
            best_score = -1
            best_idx = 0
            for i, word in enumerate(group):
                word_end = word.lower()[-3:]
                score = sum(
                    a == b for a, b in zip(reversed(imp_end), reversed(word_end))
                )
                if score > best_score:
                    best_score = score
                    best_idx = i
        replaced = group[best_idx]
        group[best_idx] = impostor.upper()
        return group, replaced

    key = next(
        (k for k in [impostor_lower, impostor.upper(), impostor.capitalize()] if k in _get_wv()),
        None
    )
    if key is None:
        idx             = random.randrange(len(group))
        replaced        = group[idx]
        group[idx]      = impostor.upper()
        return group, replaced

    best_idx, best_score = 0, -1
    for i, word in enumerate(group):
        wkey = next(
            (k for k in [word.lower(), word.upper(), word.capitalize()] if k in _get_wv()),
            None
        )
        if wkey is None: continue
        try:
            score = _get_wv().similarity(key, wkey)
            if score > best_score:
                best_score = score
                best_idx   = i
        except Exception:
            continue

    replaced        = group[best_idx]
    group[best_idx] = impostor.upper()
    return group, replaced


# ── Anchored mechanism attempts ───────────────────────────────

def try_rhyme(impostor, excl):
    """
    Build a 4-word rhyme group with impostor as anchor. Uses all CMU pronunciations
    for the impostor (merged pools) and tries many companion triples.
    """
    endings = get_rhyme_endings(impostor)
    if not endings:
        return None

    pool = []
    seen = set()
    for ending in endings:
        for w in rhyme_idx.get(ending, []):
            if w == impostor or w in excl or len(w) < 3 or is_proper_noun(w):
                continue
            if w not in seen:
                seen.add(w)
                pool.append(w)

    if len(pool) < 3:
        return None

    primary_ending = endings[0]

    def pack(companions):
        group = [impostor] + list(companions)
        if not is_valid_group(group):
            return None
        if len(set(w[-3:] for w in group)) == 1:
            return None
        return {
            'mechanism':  'rhyme',
            'ending':     primary_ending,
            'group':      [w.upper() for w in group],
            # Fixed label — avoids picking a rhyming word that is also on the board
            # (e.g. "Rhymes with VERIFY" while VERIFY is in the group).
            'connection': 'Words that rhyme',
        }

    random.shuffle(pool)
    if len(pool) <= 22:
        shuf = pool[:]
        random.shuffle(shuf)
        for companions in combinations(shuf, 3):
            r = pack(companions)
            if r:
                return r
    else:
        for _ in range(12):
            random.shuffle(pool)
            for start in range(len(pool) - 2):
                companions = tuple(pool[start : start + 3])
                r = pack(companions)
                if r:
                    return r
        for _ in range(4000):
            companions = tuple(random.sample(pool, 3))
            r = pack(companions)
            if r:
                return r
    return None


def try_anagram(impostor, excl):
    key  = get_anagram_key(impostor)
    pool = [
        w for w in anagram_idx.get(key, [])
        if w != impostor and w not in excl
        and not is_proper_noun(w)
    ]
    if len(pool) < 3: return None
    random.shuffle(pool)
    for start in range(len(pool) - 2):
        companions = pool[start:start+3]
        group      = [impostor] + companions
        if not is_valid_group(group): continue
        return {
            'mechanism':  'anagram',
            'key':        key,
            'group':      [w.upper() for w in group],
            'connection': 'Anagrams of each other',
        }
    return None


def try_letter_pattern(impostor, excl):
    impostor_lower = impostor.lower()
    matching = [p for p in LETTER_PATTERNS if p in impostor_lower and p in pattern_idx]
    if not matching: return None
    random.shuffle(matching)
    for pattern in matching:
        pool = [
            w for w in pattern_idx[pattern]
            if w != impostor_lower and w not in excl
            and not is_proper_noun(w)
        ]
        if len(pool) < 3: continue
        random.shuffle(pool)
        for start in range(min(len(pool) - 2, 30)):
            companions = pool[start:start+3]
            group      = [impostor_lower] + companions
            if not is_valid_group(group): continue
            return {
                'mechanism':  'letter_pattern',
                'pattern':    pattern,
                'group':      [w.upper() for w in group],
                'connection': f'All contain "{pattern.upper()}"',
            }
    return None


# ── Random fallback generators (no impostor) ─────────────────

def random_rhyme(excl):
    ends = list(VALID_ENDINGS.keys())
    random.shuffle(ends)
    for e in ends:
        pool = [
            w for w in VALID_ENDINGS[e]
            if w not in excl and len(w) >= 3 and not is_proper_noun(w)
        ]
        if len(pool) < 6: continue
        random.shuffle(pool)
        for s in range(len(pool) - 3):
            c = pool[s:s+4]
            if not is_valid_group(c): continue
            if len(set(w[-3:] for w in c)) == 1: continue
            return {
                'mechanism':  'rhyme',
                'ending':     e,
                'group':      [w.upper() for w in c],
                'connection': 'Words that rhyme',
            }
    return None


def random_anagram(excl):
    valid = list(VALID_ANAGRAM_GROUPS.items())
    random.shuffle(valid)
    for key, words in valid:
        pool = [w for w in words if w not in excl and not is_proper_noun(w)]
        if len(pool) < 4: continue
        random.shuffle(pool)
        for combo in combinations(pool, 4):
            if is_valid_group(list(combo)):
                return {
                    'mechanism':  'anagram',
                    'key':        key,
                    'group':      [w.upper() for w in combo],
                    'connection': 'Anagrams of each other',
                }
    return None


def random_letter_pattern(excl):
    patterns = list(pattern_idx.keys())
    random.shuffle(patterns)
    for pattern in patterns:
        pool = [
            w for w in pattern_idx[pattern]
            if w not in excl and not is_proper_noun(w)
        ]
        if len(pool) < 4: continue
        random.shuffle(pool)
        for s in range(min(len(pool) - 3, 30)):
            c = pool[s:s+4]
            if not is_valid_group(c): continue
            return {
                'mechanism':  'letter_pattern',
                'pattern':    pattern,
                'group':      [w.upper() for w in c],
                'connection': f'All contain "{pattern.upper()}"',
            }
    return None


# ── Master green generator ────────────────────────────────────

def generate_rhyme_group(existing_words=None, impostor=None) -> dict | None:
    excl           = set(w.lower() for w in (existing_words or []))
    impostor_lower = impostor.lower() if impostor else None

    if impostor:
        anchored = [
            ('rhyme',          lambda: try_rhyme(impostor_lower, excl)),
            ('anagram',        lambda: try_anagram(impostor_lower, excl)),
            ('letter_pattern', lambda: try_letter_pattern(impostor_lower, excl)),
        ]
        # 50% rhyme first, else shuffle all three
        if random.random() < 0.5:
            order = [anchored[0]] + random.sample(anchored[1:], 2)
        else:
            order = random.sample(anchored, 3)

        for name, fn in order:
            print(f'  Trying {name} anchored on "{impostor_lower}"...')
            result = fn()
            if result:
                result['has_impostor'] = True
                result['impostor']     = impostor.upper()
                result['replaced']     = None
                result['difficulty']   = 'green'
                print(f'  ✓ {name} succeeded')
                return result

        print(f'  All anchored mechanisms failed — falling back to random + inject')

    # Random fallback
    fallbacks = [
        ('rhyme',          random_rhyme),
        ('anagram',        random_anagram),
        ('letter_pattern', random_letter_pattern),
    ]
    if random.random() < 0.5:
        order = [fallbacks[0]] + random.sample(fallbacks[1:], 2)
    else:
        order = random.sample(fallbacks, 3)

    for name, fn in order:
        result = fn(excl)
        if result:
            result['difficulty']   = 'green'
            result['has_impostor'] = False

            if impostor:
                original_group = result['group'].copy()
                result['group'], replaced = inject_impostor(
                    result['group'], impostor, mechanism=result['mechanism']
                )
                result['has_impostor'] = True
                result['impostor']     = impostor.upper()
                result['replaced']     = replaced
                print(f'  Fallback inject: {replaced} → {impostor}')
                print(f'  Original       : {original_group}')

            return result

    return None


# --- Cell 9 ---
NICHE_TOPICS = [
    # Original general topics
    'things in a dentist office',
    'things on a chess board',
    'parts of a shoe',
    'things in a courtroom',
    'things that can be pitched',
    'things with strings',
    'things in a toolbox',
    'things that can be cracked',
    'types of bridges',
    'things on a ship',
    'parts of an egg',
    'things in a circus',
    'words that follow thunder',
    'things you can strike',
    'things in a hospital room',
    'parts of a camera',
    'things that can be pressed',
    'things in a barn',
    'words associated with bees',
    'things you can board',
    'parts of a bridge',
    'things that can be flat',
    'things in a magic show',
    'words that follow black',
    'things that can be drawn',
    'parts of a guitar',
    'things in a pharmacy',
    'things associated with cards',
    'things that can be rolled',
    'parts of a flower',
    'things in a kitchen drawer',
    'things that can be cast',
    'words that follow dead',
    'things associated with wine',
    'things that can be sealed',
    'parts of a clock',
    'things in a wallet',
    'things that can be tipped',
    'words that follow silver',
    'things associated with sailing',
    'things that can be topped',
    'parts of a book',
    'things in a first aid kit',
    'things that can be spotted',
    'words that follow golden',
    'things associated with boxing',
    'things that can be mixed',
    'parts of a staircase',
    'things in a garden shed',

    # Movies
    'things in a heist movie',
    'things in a horror movie',
    'things in a superhero origin story',
    'things you see in a rom-com',
    'things in a courtroom drama',
    'things in a disaster movie',
    'things in a Western',
    'things at Hogwarts',
    'things in the Batcave',
    'things on the Death Star',
    'things in a James Bond movie',
    'things in a Pixar movie',
    'things in Jurassic Park',
    'things in a spy movie',
    'things in a Rocky movie',
    'things in a mob movie',
    'things at the movies',
    'things in Oz',
    'things in the Matrix',

    # TV Shows
    'things in a sitcom apartment',
    'things on Sesame Street',
    'things in a reality TV competition',
    'things in a cooking competition show',
    'things in a true crime documentary',
    'things in a medical drama',
    'things in a game show',
    'things on Jeopardy',
    'things in a soap opera',
    'things in a talk show',
    'things in a nature documentary',
    'things at Dunder Mifflin',
    'things in Westeros',
    'things in a Law and Order episode',
    'things on Survivor',
    'things on SNL',

    # Music
    'things at a concert',
    'things in a country song',
    'things in a music video',
    'things on a tour bus',
    'things in a rock ballad',
    'things at the Grammys',
    'things at a karaoke bar',
    'things in a Broadway musical',
    'things in a jazz club',
    'things in a jukebox',
    'things at a boy band concert',

    # Sports
    'things at the Super Bowl',
    'things in a boxing ring',
    'things at a tailgate',
    'things in a locker room',
    'things at the Olympics',
    'things on a baseball diamond',
    'things at a NASCAR race',
    'things in a postgame interview',
    'things in a wrestling match',
    'things at a golf tournament',
    'things at a basketball game',
    'things in a stadium',

    # Food & Drink
    'things at a BBQ',
    'things in a diner',
    'things on a charcuterie board',
    'things at a Thanksgiving table',
    'things at a state fair',
    'things in a cocktail bar',
    'things on a brunch menu',
    'things in a pizza shop',
    'things at a coffee shop',
    'things at a potluck',

    # Games & Culture
    'things in a Mario game',
    'things on a Monopoly board',
    'things in a deck of cards',
    'things at a casino',
    'things at Comic-Con',
    'things in a crossword puzzle',
    'things at a trivia night',
    'things in a board game',
    'things at a carnival',

    # Holidays & Traditions
    'things at a Fourth of July party',
    'things in a Halloween costume store',
    'things on a Christmas tree',
    'things at a New Year\'s Eve party',
    'things in an Easter basket',
    'things at a birthday party',
    'things at a wedding reception',
    'things at a baby shower',
    'things at a graduation',
    'things at prom',

    # Celebrity & Pop Culture
    'things at a red carpet event',
    'things in a celebrity memoir',
    'things in a tabloid headline',
    'things at a movie premiere',
    'things at an awards show',
    'things in a celebrity roast',
    'things in a magazine cover story',

    # Classic Americana & General Knowledge
    'things at a circus',
    'things at a museum',
    'things in a library',
    'things at a church service',
    'things in a newspaper',
    'things at a parade',
    'things at an airport',
    'things in a hotel lobby',
    'things at a theme park',
    'things on a road trip',
    'things at a beach',
    'things in a doctor office',
    'things at summer camp',
    'things in a high school yearbook',
    'things at a funeral',
]

# --- Cell 10 ---
# ============================================================
# CELL 5 — Blue Group (LLM Niche Category)
#
# Medium-hard difficulty. No impostor in blue — it is used
# exclusively by green (rhyme anchor) and yellow (w2v seed).
# ============================================================

def verify_blue_group(group: list, connection: str) -> tuple:
    prompt = (
        f'Check this NYT Connections BLUE category (medium-hard):\n\n'
        f'Title: {connection}\n'
        f'Words: {", ".join(group)}\n\n'
        f'Does this title genuinely apply to ALL 4 words? '
        f'Be strict — if even one word is a stretch, say false.\n\n'
        f'Return only JSON:\n'
        f'{{"valid": true, "reason": "one sentence"}}'
    )
    try:
        resp   = claude_client.messages.create(
            model='claude-haiku-4-5-20251001', max_tokens=80,
            system='You are a JSON-only response bot.',
            messages=[{'role': 'user', 'content': prompt}]
        )
        result = extract_first_json(resp.content[0].text)
        return bool(result.get('valid', False)), result.get('reason', '')
    except Exception:
        return False, 'parse error'


def label_blue_group(group: list) -> str:
    """
    BLUE titles must be inferred only from the four words — not from the internal
    NICHE_TOPIC string used during generation (that would echo the scaffold).
    """
    banned = ', '.join(group)
    prompt = (
        f'Here are four words from one category in a NYT Connections puzzle '
        f'(BLUE difficulty — thoughtful, slightly lateral):\n'
        f'{", ".join(group)}\n\n'
        f'Write ONE puzzle title (max 6 words) naming what they ALL share — '
        f'the headline printed above the solved group.\n\n'
        f'Rules:\n'
        f'- Base the title ONLY on these four words — you have no other context.\n'
        f'- Sound like a real Connections editor: concrete and specific.\n'
        f'- Do NOT paste generic phrases like "types of fun", "things at a carnival" '
        f'unless those words truly nail what these four entries share.\n'
        f'- Do NOT include any of these words in the title text: {banned}\n'
        f'- Return only the title, plain text, no quotation marks wrapping the whole line.'
    )
    try:
        resp = claude_client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=48,
            messages=[{'role': 'user', 'content': prompt}],
        )
        return resp.content[0].text.strip().strip('"').strip()
    except Exception:
        return 'Related items'


def verify_yellow_group(group: list, connection: str) -> tuple:
    """YELLOW must be easy and uncontroversial — reject bogus synonym/thematic fits."""
    prompt = (
        f'This is proposed as the YELLOW (easiest) Connections group:\n\n'
        f'Title: {connection}\n'
        f'Words: {", ".join(group)}\n\n'
        f'The yellow group should be obvious to most solvers.\n'
        f'Is this title accurate for ALL four words — no weak links?\n'
        f'If the title uses "synonyms for X" or "things related to X", reject unless '
        f'EVERY word is a clear, standard fit.\n\n'
        f'Return only JSON:\n'
        f'{{"valid": true, "reason": "one sentence"}}'
    )
    try:
        resp = claude_client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=80,
            system='You are a JSON-only response bot.',
            messages=[{'role': 'user', 'content': prompt}],
        )
        result = extract_first_json(resp.content[0].text)
        return bool(result.get('valid', False)), result.get('reason', '')
    except Exception:
        return False, 'parse error'


def generate_blue_group(existing_words: list,
                         purple_conn: str = '',
                         green_conn: str = '',
                         max_attempts: int = 6) -> dict | None:
    existing_lower = set(w.lower() for w in existing_words)
    avoid_words    = ', '.join(existing_words)
    avoid_themes   = ', '.join(filter(None, [purple_conn, green_conn]))
    used_topics    = []

    for attempt in range(max_attempts):
        topic = random.choice([t for t in NICHE_TOPICS if t not in used_topics])
        used_topics.append(topic)
        print(f'  Attempt {attempt+1}: "{topic}"')

        prompt = (
            f'You are choosing four words for a NYT Connections BLUE group '
            f'(medium-hard, slightly lateral).\n\n'
            f'Internal creative brief (do NOT output this phrase as the puzzle title — '
            f'it is only for your brainstorming): {topic}\n\n'
            f'Generate exactly 4 single English words that fit that brief. '
            f'They should feel loosely related when the connection is revealed, '
            f'unrelated at first glance.\n\n'
            f'Rules:\n'
            f'- Do NOT use: {avoid_words}\n'
            f'- Do NOT relate to: {avoid_themes}\n'
            f'- Common everyday English — no proper nouns, no multi-word phrases\n'
            f'- Prefer short punchy words\n\n'
            f'Return ONLY this JSON (words only — no category title yet):\n'
            f'{{"group": ["W1","W2","W3","W4"]}}'
        )

        try:
            resp   = claude_client.messages.create(
                model='claude-haiku-4-5-20251001', max_tokens=120,
                system='You are a JSON-only response bot.',
                messages=[{'role': 'user', 'content': prompt}]
            )
            result = extract_first_json(resp.content[0].text)
            group  = [w.upper() for w in result['group']]
        except Exception as e:
            print(f'  Parse error: {e}')
            continue

        print(f'  Generated words: {group}')

        if group_has_conjugation_issues(group):
            print('  Rejected   : conjugation issue')
            continue

        overlap = [w for w in group if w.lower() in existing_lower]
        if overlap:
            print(f'  Rejected   : overlaps existing: {overlap}')
            continue

        connection = None
        verified = False
        for label_try in range(4):
            connection = label_blue_group(group)
            ok, reason = verify_blue_group(group, connection)
            print(f'  Title try {label_try + 1}: "{connection}" → verified={ok} — {reason}')
            if ok:
                verified = True
                break
        if not verified:
            print('  Rejected   : could not verify title')
            continue

        return {
            'mechanism':   'llm_niche',
            'topic':       topic,
            'group':       group,
            'connection':  connection,
            'difficulty':  'blue',
            'has_impostor': False,
        }

    return None


# --- Cell 11 ---
# ============================================================
# CELL 6 — Yellow Group (word2vec, Easy)
#
# If impostor_result exists, imp2 is used as the word2vec
# seed — the other 3 words cluster naturally around it.
# The impostor sits as a natural member of the group.
# ============================================================

def find_similar_w2v(anchor, existing_words, topn=200, n=3):
    al   = anchor.lower()
    excl = set(w.lower() for w in existing_words)
    key  = next((k for k in [al, anchor.upper(), anchor.capitalize()] if k in _get_wv()), None)
    if key is None:
        return []
    try:
        ap = pos_tag([al])[0][1]
    except:
        ap = None
    res = []
    for word, _ in _get_wv().most_similar(key, topn=topn):
        word = word.lower().strip()
        if '_' in word or word not in COMMON_WORDS or word not in FULL_DICT:
            continue
        if word in excl or word == al:
            continue
        if lev(al, word) / max(len(al), len(word)) < 0.3:
            continue
        if word in al or al in word:
            continue
        if get_lemmas(al) & get_lemmas(word):
            continue
        if ap:
            try:
                if pos_tag([word])[0][1][0] != ap[0]:
                    continue
            except:
                pass
        res.append(word)
        if len(res) == n:
            break
    return res


def label_yellow_group(group: list) -> str:
    words_joined = ', '.join(group)
    banned_in_label = ', '.join(group)
    prompt = (
        f'These 4 words are the YELLOW (easiest) group in NYT Connections: '
        f'{words_joined}.\n\n'
        f'Write ONE connection title (max 6 words) — a simple NYT-style category name.\n\n'
        f'Rules:\n'
        f'- The title must fit ALL four words equally — no stretch.\n'
        f'- Avoid "Synonyms for \\"...\\"" unless each word is a straight dictionary '
        f'synonym of that headword; if the fit is loose, use a broader theme instead '
        f'(e.g. journey, conflict, effort) or a different framing entirely.\n'
        f'- Prefer concrete categories: "Types of ___", "Things in a ___", '
        f'"Ways to ___", "Parts of a ___", fill-in-the-blank titles.\n'
        f'- Do NOT use follow/precede patterns unless every word clearly works.\n'
        f'- Do NOT include any of these words in the title: {banned_in_label}\n'
        f'- Do NOT use vague "Words associated with" / "Things related to".\n\n'
        f'Return only the title, plain text.'
    )
    try:
        resp = claude_client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=48,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return resp.content[0].text.strip().strip('"')
    except Exception:
        return f'Words related to {group[0]}'


def yellow_connection_with_retries(group: list, max_tries: int = 4):
    """LLM title + verifier — rejects sloppy synonym/thematic labels."""
    for attempt in range(max_tries):
        label = label_yellow_group(group)
        ok, reason = verify_yellow_group(group, label)
        print(f'    yellow title try {attempt + 1}: "{label}" → {ok} — {reason}')
        if ok:
            return label
    return None


def generate_yellow_group(existing_words, impostor=None, max_attempts=40):
    excl  = set(w.lower() for w in existing_words)
    tried = set()

    if impostor:
        impostor_lower = impostor.lower()
        # Use impostor as seed — exclude it from the similar pool
        # so the 3 companions don't include it, then add it back
        excl_with_imp = excl | {impostor_lower}
        sim = find_similar_w2v(impostor_lower, list(excl_with_imp) + list(tried), n=3)
        if len(sim) == 3:
            group = [impostor.upper()] + [w.upper() for w in sim]
            if not group_has_conjugation_issues(group):
                label = yellow_connection_with_retries(group)
                if label:
                    return {
                        'mechanism':   'word2vec',
                        'seed':        impostor.upper(),
                        'group':       group,
                        'connection':  label,
                        'difficulty':  'yellow',
                        'has_impostor': True,
                        'impostor':    impostor.upper(),
                        'replaced':    None,
                    }
        print(f'  imp2 w2v seed failed for "{impostor}" — falling back to random seed')

    # Normal generation
    shorts = [
        w for w in SHORT_WORDS
        if w not in excl
        and any(k in _get_wv() for k in [w, w.upper(), w.capitalize()])
    ]
    longs = [
        w for w in COMMON_WORDS
        if len(w) > 6 and w not in excl
        and any(k in _get_wv() for k in [w, w.upper(), w.capitalize()])
    ]
    random.shuffle(shorts)
    random.shuffle(longs)

    n_short = int(max_attempts * 0.6)
    seeds   = shorts[:n_short] + longs[:max_attempts - n_short]

    for seed in seeds:
        if seed in tried:
            continue
        tried.add(seed)
        sim = find_similar_w2v(seed, list(excl) + list(tried))
        if len(sim) != 3:
            continue
        group = [seed.upper()] + [w.upper() for w in sim]
        if group_has_conjugation_issues(group):
            continue
        label = yellow_connection_with_retries(group)
        if not label:
            continue
        return {
            'mechanism':   'word2vec',
            'seed':        seed.upper(),
            'group':       group,
            'connection':  label,
            'difficulty':  'yellow',
            'has_impostor': False,
        }

    return None


# ============================================================
# Entry point for BurakAdapter in master_generator
# ============================================================

def run_full_pipeline() -> dict | None:
    """
    Runs the full puzzle generation pipeline.
    Called by BurakAdapter.generate().
    Returns puzzle dict with yellow/green/blue/purple keys, or None.
    """
    used: list[str] = []

    # Purple + impostors
    purple = generate_purple_group(n_candidates=8)
    if not purple:
        return None
    used += purple['group']

    impostor_result = select_two_impostors(purple['group'])
    if impostor_result:
        import random as _random
        targets = _random.choices(
            [('blue', 'yellow'), ('blue', 'green'), ('green', 'yellow')],
            weights=[50, 25, 25]
        )[0]
        impostor_result['imp1_target'] = targets[0]
        impostor_result['imp2_target'] = targets[1]

    imp_for_green  = impostor_result['imp1'] if impostor_result else None
    imp_for_yellow = impostor_result['imp2'] if impostor_result else None

    # Green
    green = generate_rhyme_group(existing_words=used, impostor=imp_for_green)
    if not green:
        return None
    used += green['group']

    # Blue
    blue = generate_blue_group(
        existing_words=used,
        purple_conn=purple.get('connection', ''),
        green_conn=green.get('connection', '')
    )
    if not blue:
        return None
    used += blue['group']

    # Yellow
    yellow = generate_yellow_group(existing_words=used, impostor=imp_for_yellow)
    if not yellow:
        return None
    used += yellow['group']

    return {
        'yellow':          yellow,
        'green':           green,
        'blue':            blue,
        'purple':          purple,
        'impostor_result': impostor_result,
    }


if __name__ == '__main__':
    puzzle = run_full_pipeline()
    if puzzle is None:
        print('[burak_pipeline] run_full_pipeline() returned None')
        raise SystemExit(1)
    print('[burak_pipeline] OK — puzzle generated.')
    for colour in ('yellow', 'green', 'blue', 'purple'):
        g = puzzle[colour]
        conn = g.get('connection', '')[:60]
        print(f'  {colour:6}  {conn}  {g.get("group")}')