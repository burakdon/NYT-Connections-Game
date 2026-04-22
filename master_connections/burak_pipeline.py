# ============================================================
# burak_pipeline.py
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

from collections import Counter, defaultdict, deque
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

# Ban words that cause bad rhyme groups or visual mismatches
COMMON_WORDS -= {'corp', 'corps', 'genre', 'cache'}

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


def _purple_selection_mode() -> str:
    """
    pass_random (default): score with classifier; pick uniformly among
        candidates with score >= threshold; if none pass, use highest score.
    best: always pick highest classifier score (legacy behavior).
    ignore: do not call the classifier; pick uniformly among collected candidates.
    """
    v = (os.environ.get('BURAK_PURPLE_SELECTION') or 'pass_random').strip().lower()
    if v in ('best', 'top', 'max'):
        return 'best'
    if v in ('ignore', 'none', 'off', 'no_classifier', 'random', 'skip_classifier'):
        return 'ignore'
    return 'pass_random'


def _purple_pass_threshold() -> float:
    try:
        return float(os.environ.get('BURAK_PURPLE_CLASSIFIER_THRESHOLD', '0.7'))
    except Exception:
        return 0.7


def generate_purple_group(n_candidates=8):
    """
    Try shuffled mechanisms, collect up to ``n_candidates`` purple groups.

    Selection (``BURAK_PURPLE_SELECTION``):
      - ``pass_random`` (default): ML score each; choose uniformly at random
        among scores >= ``BURAK_PURPLE_CLASSIFIER_THRESHOLD`` (default 0.7);
        if none pass, fall back to the single highest score.
      - ``best``: always the highest score (old behavior — often repeats roots).
      - ``ignore``: skip the classifier; uniform random among collected candidates.
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

    mode = _purple_selection_mode()
    threshold = _purple_pass_threshold()

    candidates = []
    for name, fn in generators:
        r = fn()
        if r is None:
            print(f'  [{name}] no candidate')
            continue
        if mode == 'ignore':
            print(f'  [{name}] (classifier skipped) {r["group"]}')
            candidates.append((0.0, r))
        else:
            score = score_candidate(r)
            tag   = 'keep' if score >= 0.8 else ('borderline' if score >= 0.3 else 'reject')
            print(f'  [{name}] score={score:.3f} ({tag}) {r["group"]}')
            candidates.append((score, r))
        if len(candidates) >= n_candidates:
            break

    if not candidates:
        return None

    if mode == 'ignore':
        best = random.choice([c[1] for c in candidates])
        print(f'\n  Chosen (random, no classifier): [{best["mechanism"]}] {best["group"]}')
        return best

    if mode == 'best':
        candidates.sort(key=lambda x: -x[0])
        best_score, best = candidates[0]
        print(f'\n  Best: [{best["mechanism"]}] score={best_score:.3f}')
        return best

    passing = [(s, r) for s, r in candidates if s >= threshold]
    if passing:
        best_score, best = random.choice(passing)
        print(
            f'\n  Chosen: [{best["mechanism"]}] score={best_score:.3f} '
            f'(random among {len(passing)} with score ≥ {threshold})'
        )
    else:
        candidates.sort(key=lambda x: -x[0])
        best_score, best = candidates[0]
        print(
            f'\n  Chosen (fallback — best score): [{best["mechanism"]}] '
            f'score={best_score:.3f} (none ≥ {threshold})'
        )
    return best

print('Scoring + master generator ready.')

# Keep a short memory of recent imp2 seeds across pipeline calls to reduce
# repetitive yellow anchors like repeated WOOD/OFF/etc on adjacent retries.
RECENT_IMP2_SEEDS = deque(maxlen=48)


def _imp2_recent_cooldown() -> int:
    try:
        return max(0, min(40, int(os.environ.get('BURAK_IMP2_RECENT_COOLDOWN', '10'))))
    except Exception:
        return 10

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


def find_impostor_green(purple_word: str,
                         purple_group: list,
                         exclude_extra: list = [],
                         min_rhyme_pool: int = 4,
                         min_anagram_pool: int = 3,
                         min_pattern_pool: int = 4) -> tuple[str | None, str | None]:
    """
    Find an impostor that can anchor ANY green mechanism:
    rhyme, anagram, or letter pattern.
    Semantically similar to purple_word via word2vec.
    Accepts the first word that qualifies for any mechanism.
    """
    anchor_lower = purple_word.lower()
    excluded     = set(w.lower() for w in purple_group + exclude_extra)

    key = next(
        (k for k in [anchor_lower, purple_word.upper(), purple_word.capitalize()]
         if k in _get_wv()),
        None
    )
    if key is None:
        return None, None

    for word, _ in _get_wv().most_similar(key, topn=500):
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

        # Check 1 — rhyme capability
        ending = get_rhyme_ending(word)
        if ending:
            full_pool = rhyme_idx.get(ending, [])
            # Require larger pool for rare endings to avoid thin-pool traps
            min_pool = min_rhyme_pool if len(full_pool) >= 10 else min_rhyme_pool + 4
            rhyme_pool = [
                w for w in full_pool
                if w != word and w not in excluded
                and len(w) >= 3 and not is_proper_noun(w)
            ]
            if len(rhyme_pool) >= min_pool:
                return word, 'rhyme'

        # Check 2 — anagram capability
        ana_key  = get_anagram_key(word)
        ana_pool = [
            w for w in anagram_idx.get(ana_key, [])
            if w != word and w not in excluded
            and not is_proper_noun(w)
        ]
        if len(ana_pool) >= min_anagram_pool:
            return word, 'anagram'

        # Check 3 — anchor-derived meaningful short-word substring capability.
        if _word_substring_can_build_from_anchor(word, excluded, min_pattern_pool):
            return word, 'letter_pattern'

        # Check 3 — letter pattern capability (substring-style before suffix-style)
        for pattern in _letter_pattern_try_order():
            kind = pattern_kind.get(pattern, 'substring')
            if kind == 'suffix':
                if not word.endswith(pattern):
                    continue
            elif pattern not in word:
                continue
            pat_pool = [
                w for w in pattern_idx[pattern]
                if w != word and w not in excluded
                and not is_proper_noun(w)
                and len(w) <= 10
            ]
            if len(pat_pool) >= min_pattern_pool:
                return word, 'letter_pattern'

    return None, None


def find_impostor_w2v_seed(purple_word: str,
                             purple_group: list,
                             exclude_extra: list = []) -> tuple[str, str] | tuple[None, None]:
    """
    Find an impostor that:
      - Is semantically similar to purple_word (word2vec)
      - Can seed yellow the same way ``generate_yellow_group`` will: three
        ``find_similar_w2v(..., pos_filter=False)`` neighbors plus passing
        ``group_has_conjugation_issues``.

    ``exclude_extra`` should include every word already committed to the board
    for this puzzle (e.g. ``imp1`` and, after green exists, all four green words)
    so feasibility matches ``generate_yellow_group``.

    Returns (word, 'yellow') or (None, None).
    """
    anchor_lower = purple_word.lower()
    excluded     = set(w.lower() for w in purple_group + list(exclude_extra))

    key = next(
        (k for k in [anchor_lower, purple_word.upper(), purple_word.capitalize()]
         if k in _get_wv()),
        None
    )
    if key is None:
        return None, None
    recent_block = set(list(RECENT_IMP2_SEEDS)[-_imp2_recent_cooldown() :])

    for word, _ in _get_wv().most_similar(key, topn=500):
        word = word.lower().strip()
        if '_' in word:
            continue
        if word not in COMMON_WORDS or word not in FULL_DICT:
            continue
        if word in excluded:
            continue
        if word in recent_block:
            continue
        if lev(anchor_lower, word) / max(len(anchor_lower), len(word)) < 0.3:
            continue
        if word in anchor_lower or anchor_lower in word:
            continue
        if get_lemmas(word) & get_lemmas(anchor_lower):
            continue
        if any(get_lemmas(word) & get_lemmas(p.lower()) for p in purple_group):
            continue

        # Same bar as generate_yellow_group(impostor=...): w2v neighbors without
        # POS filtering, plus conjugation validity on the resulting foursome.
        if _imp2_can_build_yellow_group(word, excluded):
            RECENT_IMP2_SEEDS.append(word)
            return word, 'yellow'

    return None, None


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
# No trivial 2-letter clusters (e.g. "CK") — use morpheme-like suffixes or
# substantive multigraphs (3+ letters) so green "letter" groups read as wordplay.
#
# Default suffix set deliberately omits -tion / -sion (overused "four long Latin words").
# Set BURAK_GREEN_LETTER_ALLOW_TION=1 to add them back to the index.

def _letter_suffix_patterns_for_index() -> tuple[str, ...]:
    core = (
        'ness', 'less', 'ment', 'able', 'ible',
        'ful', 'ous', 'ive', 'ship', 'hood', 'ward', 'wise',
    )
    if (os.environ.get('BURAK_GREEN_LETTER_ALLOW_TION') or '').strip().lower() in (
        '1', 'true', 'yes', 'on',
    ):
        return core + ('tion', 'sion')
    return core

LETTER_SUBSTRING_PATTERNS = (
    'ght', 'tch', 'dge', 'nch', 'rth', 'lth',
    'scr', 'shr', 'spl', 'spr', 'str', 'thr',
    'ough', 'augh', 'eigh', 'igh',
)

MEANINGFUL_SUBSTRING_BLOCKLIST = {
    'and', 'the', 'for', 'you', 'are', 'but', 'not', 'all', 'any', 'can', 'our', 'out', 'off',
    'from', 'with', 'this', 'that', 'have', 'had', 'was', 'were', 'who', 'why', 'how', 'its',
    'into', 'onto', 'than', 'then', 'them', 'they', 'their', 'there', 'where', 'when', 'what',
    'your', 'just', 'very', 'only', 'much', 'many', 'more', 'most', 'less', 'some',
}


def _letter_pattern_word_ok(w: str, pattern: str) -> bool:
    wl = w.lower()
    return (
        wl in FULL_DICT
        and len(wl) >= 4
        and not is_proper_noun(w)
        and len(wl) <= 12
    )


def _meaningful_substring_pool_bounds() -> tuple[int, int]:
    try:
        min_pool = int(os.environ.get('BURAK_GREEN_WORD_SUBSTRING_MIN_POOL', '6'))
    except Exception:
        min_pool = 6
    try:
        max_pool = int(os.environ.get('BURAK_GREEN_WORD_SUBSTRING_MAX_POOL', '40'))
    except Exception:
        max_pool = 40
    min_pool = max(4, min(30, min_pool))
    max_pool = max(min_pool + 2, min(120, max_pool))
    return min_pool, max_pool


def _meaningful_substring_tokens() -> list[str]:
    """
    Candidate substrings that are themselves short content words (AIR, HAND, FIRE...).
    Excludes function words to avoid mushy "contains AND/THE/ARE" categories.
    """
    toks = []
    for w in SHORT_WORDS:
        wl = w.lower()
        if len(wl) < 3 or len(wl) > 5:
            continue
        if wl in MEANINGFUL_SUBSTRING_BLOCKLIST:
            continue
        if not wl.isalpha():
            continue
        if wl not in FULL_DICT or is_proper_noun(wl):
            continue
        toks.append(wl)
    toks.sort(key=lambda t: (len(t), t))
    return toks


def _contains_interior_token(word: str, token: str) -> bool:
    wl = word.lower()
    tl = token.lower()
    i = wl.find(tl)
    return i > 0 and (i + len(tl) < len(wl))


def _anchor_meaningful_tokens(anchor: str) -> list[str]:
    """
    Real short words found strictly inside the anchor (not prefix/suffix).
    Example: architecture -> tech, hand, etc. (if valid in SHORT_WORDS).
    """
    a = anchor.lower().strip()
    out = set()
    for n in (5, 4, 3):
        if len(a) <= n + 1:
            continue
        for i in range(1, len(a) - n):
            tok = a[i : i + n]
            if not tok.isalpha():
                continue
            if tok in MEANINGFUL_SUBSTRING_BLOCKLIST:
                continue
            if tok not in SHORT_WORDS or tok not in FULL_DICT:
                continue
            if is_proper_noun(tok):
                continue
            out.add(tok)
    return sorted(out, key=lambda t: (-len(t), t))


def _anchor_word_substring_pool(token: str, excluded: set, max_word_len: int = 10) -> list[str]:
    return [
        w for w in COMMON_WORDS
        if w not in excluded
        and token in w
        and _letter_pattern_word_ok(w, token)
        and len(w) <= max_word_len
    ]


def _word_substring_can_build_from_anchor(
    anchor: str, excluded: set, min_pattern_pool: int = 4
) -> bool:
    for tok in _anchor_meaningful_tokens(anchor):
        pool = _anchor_word_substring_pool(tok, excluded, max_word_len=10)
        if len(pool) < min_pattern_pool:
            continue
        # Require at least one companion where token appears internally too.
        if any(_contains_interior_token(w, tok) for w in pool):
            return True
    return False


def letter_pattern_connection(pattern: str, kind: str) -> str:
    """NYT-style wording by pattern kind."""
    pu = pattern.upper()
    if kind == 'suffix':
        return f'Words ending in "{pu}"'
    if kind == 'word_substring_anchor':
        return f'Contain hidden "{pu}"'
    if kind == 'word_substring':
        return f'All contain "{pu}"'
    return f'All contain "{pu}"'


print('Building letter pattern index...')
pattern_idx: dict[str, list] = {}
pattern_kind: dict[str, str] = {}  # 'suffix' | 'substring' | 'word_substring'

for pattern in _letter_suffix_patterns_for_index():
    pool = [
        w for w in COMMON_WORDS
        if w.endswith(pattern) and _letter_pattern_word_ok(w, pattern)
    ]
    if len(pool) >= 8:
        pattern_idx[pattern] = pool
        pattern_kind[pattern] = 'suffix'

for pattern in LETTER_SUBSTRING_PATTERNS:
    if pattern in pattern_idx:
        continue
    pool = [
        w for w in COMMON_WORDS
        if pattern in w.lower() and _letter_pattern_word_ok(w, pattern)
    ]
    if len(pool) >= 8:
        pattern_idx[pattern] = pool
        pattern_kind[pattern] = 'substring'

ms_min, ms_max = _meaningful_substring_pool_bounds()
for token in _meaningful_substring_tokens():
    if token in pattern_idx:
        continue
    pool = [
        w for w in COMMON_WORDS
        if token in w.lower() and _letter_pattern_word_ok(w, token)
    ]
    # Avoid both sparse and overly broad tokens (e.g. tiny glue-like fragments).
    if len(pool) < ms_min or len(pool) > ms_max:
        continue
    pattern_idx[token] = pool
    pattern_kind[token] = 'word_substring'

print(f'Valid letter patterns : {len(pattern_idx)}')


def _letter_pattern_try_order() -> list[str]:
    """
    Prefer meaningful short-word substrings, then multigraph patterns, then suffixes
    when scanning or building groups — improves variety vs. "-tion" homework sets.
    """
    words = [p for p in pattern_idx if pattern_kind.get(p) == 'word_substring']
    subs = [p for p in pattern_idx if pattern_kind.get(p) == 'substring']
    sufs = [p for p in pattern_idx if pattern_kind.get(p) == 'suffix']
    random.shuffle(words)
    random.shuffle(subs)
    random.shuffle(sufs)
    return words + subs + sufs


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
        if imp_set:
            for i, we in enumerate(word_ends):
                if we and we not in imp_set:
                    best_idx = i
                    break
        if best_idx is None:
            cnt = Counter(e for e in word_ends if e)
            ref = cnt.most_common(1)[0][0] if cnt else None
            if ref:
                for i, we in enumerate(word_ends):
                    if we != ref:
                        best_idx = i
                        break
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
    Build a 4-word rhyme group with impostor as anchor.
    Tries each CMU pronunciation ending as the rhyme key in turn.
    """
    endings = get_rhyme_endings(impostor)
    if not endings:
        return None

    n_random = _rhyme_anchor_random_trials()
    endings_shuf = endings[:]
    random.shuffle(endings_shuf)

    def pack(primary_ending, companions):
        group = [impostor] + list(companions)
        if not is_valid_group(group):
            return None
        if len(set(w[-3:] for w in group)) == 1:
            return None
        if not all(primary_ending in get_rhyme_endings(w) for w in group):
            return None
        group_lower = [w.lower() for w in [impostor] + list(companions)]
        if not all(w in CMU for w in group_lower):
            return None
        return {
            'mechanism':  'rhyme',
            'ending':     primary_ending,
            'group':      [w.upper() for w in group],
            'connection': 'Words that rhyme',
        }

    def search_pool(primary_ending, pool):
        if len(pool) < 3:
            return None
        random.shuffle(pool)
        if len(pool) <= 22:
            shuf = pool[:]
            random.shuffle(shuf)
            for companions in combinations(shuf, 3):
                r = pack(primary_ending, companions)
                if r:
                    return r
            return None
        for _ in range(12):
            random.shuffle(pool)
            for start in range(max(0, len(pool) - 2)):
                companions = tuple(pool[start : start + 3])
                r = pack(primary_ending, companions)
                if r:
                    return r
        for _ in range(n_random):
            companions = tuple(random.sample(pool, 3))
            r = pack(primary_ending, companions)
            if r:
                return r
        return None

    for primary_ending in endings_shuf:
        pool = [
            w
            for w in rhyme_idx.get(primary_ending, [])
            if w != impostor
            and w not in excl
            and len(w) >= 3
            and not is_proper_noun(w)
            and primary_ending in get_rhyme_endings(w)
        ]
        hit = search_pool(primary_ending, pool)
        if hit:
            return hit
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
    # First: anchor-derived meaningful short word inside impostor (interior substring).
    for token in _anchor_meaningful_tokens(impostor_lower):
        pool = _anchor_word_substring_pool(token, excl, max_word_len=10)
        if len(pool) < 3:
            continue
        interior = [w for w in pool if _contains_interior_token(w, token)]
        edge = [w for w in pool if w not in interior]
        random.shuffle(interior)
        random.shuffle(edge)
        ordered = interior + edge
        for start in range(min(len(ordered) - 2, 40)):
            companions = ordered[start:start + 3]
            group = [impostor_lower] + companions
            if not is_valid_group(group):
                continue
            # Anchor is interior by construction; demand >=1 companion interior too.
            interior_count = sum(1 for w in group if _contains_interior_token(w, token))
            if interior_count < 2:
                continue
            return {
                'mechanism':     'letter_pattern',
                'pattern':       token,
                'pattern_kind':  'word_substring_anchor',
                'group':         [w.upper() for w in group],
                'connection':    letter_pattern_connection(token, 'word_substring_anchor'),
            }

    matching = []
    for p in pattern_idx:
        k = pattern_kind.get(p, 'substring')
        if k == 'suffix':
            if impostor_lower.endswith(p):
                matching.append(p)
        elif p in impostor_lower:
            matching.append(p)
    if not matching:
        return None
    sub_m = [p for p in matching if pattern_kind.get(p) == 'substring']
    suf_m = [p for p in matching if pattern_kind.get(p) == 'suffix']
    random.shuffle(sub_m)
    random.shuffle(suf_m)
    matching = sub_m + suf_m
    for pattern in matching:
        kind = pattern_kind.get(pattern, 'substring')
        pool = [
            w for w in pattern_idx[pattern]
            if w != impostor_lower and w not in excl
            and not is_proper_noun(w)
            and len(w) <= 10
        ]
        if len(pool) < 3: continue
        random.shuffle(pool)
        for start in range(min(len(pool) - 2, 30)):
            companions = pool[start:start+3]
            group      = [impostor_lower] + companions
            if not is_valid_group(group): continue
            return {
                'mechanism':     'letter_pattern',
                'pattern':       pattern,
                'pattern_kind':  kind,
                'group':         [w.upper() for w in group],
                'connection':    letter_pattern_connection(pattern, kind),
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
    for pattern in _letter_pattern_try_order():
        pool = [
            w for w in pattern_idx[pattern]
            if w not in excl and not is_proper_noun(w)
            and len(w) <= 10
        ]
        if len(pool) < 4: continue
        random.shuffle(pool)
        for s in range(min(len(pool) - 3, 30)):
            c = pool[s:s+4]
            if not is_valid_group(c): continue
            kind = pattern_kind.get(pattern, 'substring')
            return {
                'mechanism':     'letter_pattern',
                'pattern':       pattern,
                'pattern_kind':  kind,
                'group':         [w.upper() for w in c],
                'connection':    letter_pattern_connection(pattern, kind),
            }
    return None


# ── Green mechanism integrity check ───────────────────────────

def _green_allow_impostor_inject() -> bool:
    v = (os.environ.get('BURAK_GREEN_ALLOW_IMPOSTOR_INJECT') or '').strip().lower()
    return v in ('1', 'true', 'yes', 'on')


def _rhyme_anchor_random_trials() -> int:
    try:
        return max(2000, min(100_000, int(os.environ.get('BURAK_RHYME_ANCHOR_TRIALS', '20000'))))
    except Exception:
        return 20_000


def _common_rhyme_ending_for_group(group: list) -> str | None:
    sets = [set(get_rhyme_endings(str(w).lower())) for w in group]
    if not all(sets):
        return None
    common = sets[0].copy()
    for s in sets[1:]:
        common &= s
    if not common:
        return None
    return sorted(common, key=len)[0]


def green_mechanism_holds(result: dict) -> bool:
    """
    Ensure the displayed green connection still matches the final words.
    """
    mech = (result.get('mechanism') or '').lower().strip()
    group = [str(w).lower() for w in (result.get('group') or [])]
    if len(group) != 4:
        return False

    if mech == 'letter_pattern':
        pattern = (result.get('pattern') or '').lower().strip()
        if not pattern:
            return False
        kind = (result.get('pattern_kind') or '').lower().strip()
        if not kind:
            kind = pattern_kind.get(pattern, 'substring')
        if kind == 'suffix':
            return all(w.endswith(pattern) for w in group)
        if kind == 'word_substring_anchor':
            return all(pattern in w for w in group)
        return all(pattern in w for w in group)

    if mech == 'anagram':
        key = result.get('key')
        return bool(key) and all(get_anagram_key(w) == key for w in group)

    if mech == 'rhyme':
        ending = result.get('ending')
        if not ending:
            return False
        return all(ending in get_rhyme_endings(w) for w in group)

    return True


# ── Master green generator ────────────────────────────────────

def generate_rhyme_group(existing_words=None, impostor=None, impostor_result=None) -> dict | None:
    excl           = set(w.lower() for w in (existing_words or []))
    impostor_lower = impostor.lower() if impostor else None

    if impostor:
        anchored = {
            'rhyme':          ('rhyme',          lambda: try_rhyme(impostor_lower, excl)),
            'anagram':        ('anagram',        lambda: try_anagram(impostor_lower, excl)),
            'letter_pattern': ('letter_pattern', lambda: try_letter_pattern(impostor_lower, excl)),
        }

        # Use the mechanism imp1 qualified for as first priority
        # Fall back to random ordering of the remaining two
        preferred = impostor_result.get('imp1_mechanism') if impostor_result else None
        if preferred and preferred in anchored:
            first   = anchored[preferred]
            rest    = [v for k, v in anchored.items() if k != preferred]
            random.shuffle(rest)
            order   = [first] + rest
        else:
            order   = list(anchored.values())
            random.shuffle(order)

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

        # All anchored mechanisms failed — fall back to green without impostor
        # A correct puzzle is more important than forcing the impostor mechanic
        print('  All anchored mechanisms failed — generating green without impostor')

    # Random fallback (no impostor): favor rhyme + anagram over suffix-style letter groups.
    fallbacks = [
        ('rhyme',          random_rhyme),
        ('anagram',        random_anagram),
        ('letter_pattern', random_letter_pattern),
    ]
    if random.random() < 0.62:
        first_two = [('rhyme', random_rhyme), ('anagram', random_anagram)]
        random.shuffle(first_two)
        order = first_two + [('letter_pattern', random_letter_pattern)]
    else:
        order = random.sample(fallbacks, 3)

    for name, fn in order:
        result = fn(excl)
        if result:
            result['difficulty']   = 'green'
            result['has_impostor'] = False

            if result.get('mechanism') == 'rhyme':
                ce = _common_rhyme_ending_for_group(result['group'])
                if ce:
                    result['ending'] = ce

            if not green_mechanism_holds(result):
                print(
                    f'  Rejecting green group: mechanism "{result.get("mechanism")}" '
                    f'no longer matches words {result.get("group")}'
                )
                continue

            # Extra guard: for rhyme groups, every word must be in CMU dict
            if result.get('mechanism') == 'rhyme':
                group_lower = [w.lower() for w in result['group']]
                if not all(w in CMU for w in group_lower):
                    missing = [w for w in group_lower if w not in CMU]
                    print(f'  Rejecting green group: words not in CMU dict: {missing}')
                    continue

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
    # Extra niche lines (expanded variety)
    'things in a planetarium',
    'things in a hardware store',
    'things in a ceramics studio',
    'things in a sewing kit',
    'things on a camping trip',
    'things in a greenhouse',
    'things in a recording studio',
    'things in a radio station',
    'things in a newsroom',
    'things in a tattoo parlor',
    'things in a mechanic garage',
    'things in a bike shop',
    'things in a barber shop',
    'things in a nail salon',
    'things in a yoga studio',
    'things in a ballet class',
    'things in an improv class',
    'things in a pottery wheel setup',
    'things in a courtroom hallway',
    'things in a city council meeting',
    'things in a polling place',
    'things in a campaign office',
    'things in a weather forecast',
    'things in a rocket launch',
    'things in mission control',
    'things in a submarine',
    'things in a lighthouse',
    'things in a train station',
    'things in a subway car',
    'things in an airport security line',
    'things in a passport office',
    'things in a hotel minibar',
    'things in a ski lodge',
    'things at a county fair',
    'things at a farmers market',
    'things at a flea market',
    'things in an antique shop',
    'things in a vintage record store',
    'things in a comic book shop',
    'things in a toy store',
    'things in a pet store',
    'things in an aquarium',
    'things in a zoo enclosure',
    'things in a bird sanctuary',
    'things in a horse stable',
    'things in a beehive setup',
    'things in a fishing tackle box',
    'things on a sailboat',
    'things in a marina',
    'things in a scuba dive',
    'things on a hiking trail',
    'things in a mountain cabin',
    'things in a desert campsite',
    'things in a rainforest documentary',
    'things in a volcano documentary',
    'things in an archaeology dig',
    'things in an art museum gift shop',
    'things in a science fair',
    'things in a chemistry lab',
    'things in a biology lab',
    'things in a computer lab',
    'things in a robotics workshop',
    'things in a chess tournament',
    'things in a poker game',
    'things in an escape room',
    'things in a haunted house attraction',
    'things in a magic club meeting',
    'things at a book signing',
    'things in a publishing house',
    'things in a writer room',
    'things in a podcast studio',
    'things in a startup pitch deck',
    'things in a board meeting',
    'things in a stock trading floor',
    'things in a shipping warehouse',
    'things in a grocery checkout lane',
    'things in a bakery display case',
    'things in an ice cream shop',
    'things in a sushi bar',
    'things in a taco truck',
    'things in a ramen shop',
    'things in a fine dining kitchen',
    'things in a brunch cafe',
    'things in a food court',
    'things in a midnight diner',
    'things in a speakeasy',
    'things in a cocktail shaker set',
    'things in a tea house',
    'things in a chocolate factory',
    'things in a perfume shop',
    'things in a flower market',
    'things in a wedding planner kit',
    'things in a baby nursery',
    'things in a moving truck',
    'things in a dorm room',
    'things in a garage sale',
    'things in a neighborhood block party',
    'things in a street parade float',
    # Additional enrichment set
    'things in a newsroom control room',
    'things in a courthouse clerk office',
    'things in a fire station',
    'things in a police dispatch center',
    'things in an emergency room waiting area',
    'things in an ambulance',
    'things in a pharmacist station',
    'things in a blood donation center',
    'things in a dentist waiting room',
    'things in an optometrist office',
    'things in a rehearsal studio',
    'things in a film set trailer',
    'things in a costume department',
    'things in a prop warehouse',
    'things in a theater backstage area',
    'things in an orchestra pit',
    'things in a choir rehearsal',
    'things in a stand-up comedy club',
    'things in a late night writers room',
    'things in a radio call-in show',
    'things in a weather station',
    'things in a satellite image',
    'things in a drone kit',
    'things in a 3D printer workshop',
    'things in a woodworking shop',
    'things in a metalworking shop',
    'things in a glassblowing studio',
    'things in a printmaking studio',
    'things in a darkroom',
    'things in a paintball arena',
    'things in a skate park',
    'things in a climbing gym',
    'things in a martial arts dojo',
    'things in a fencing bout',
    'things in a rowing regatta',
    'things in a marathon race packet',
    'things in a triathlon transition area',
    'things in a baseball bullpen',
    'things in a soccer locker room',
    'things in a tennis match',
    'things in a hockey rink',
    'things in a curling match',
    'things in a bowling alley',
    'things in a billiards hall',
    'things in a dart league',
    'things in a bridge club',
    'things in a puzzle hunt',
    'things in a geocaching kit',
    'things in a detective board',
    'things in a true crime podcast',
    'things in a genealogy archive',
    'things in a museum restoration lab',
    'things in a natural history exhibit',
    'things in a planetarium show',
    'things in a space museum',
    'things in a train conductor bag',
    'things in a ferry terminal',
    'things in a bus depot',
    'things in a road toll booth',
    'things in a border crossing',
    'things in a customs declaration',
    'things in a map room',
    'things in a travel agency',
    'things in a campsite cooler',
    'things in a tackle shop',
    'things in a hunting lodge',
    'things in a vineyard',
    'things in a brewery',
    'things in a distillery',
    'things in a farmers co-op',
    'things in a butcher shop',
    'things in a seafood market',
    'things in a spice shop',
    'things in an olive oil bar',
    'things in a candy store',
    'things in a donut shop',
    'things in a bodega',
    'things in a corner store',
    'things in a mall kiosk',
    'things in a dry cleaner',
    'things in a laundromat',
    'things in a tailor shop',
    'things in a thrift store',
    'things in a pawn shop',
    'things in a storage unit',
    'things in a moving checklist',
    'things in a home inspection report',
    'things in a real estate listing',
    'things in a mortgage office',
    'things in a bank vault',
    'things in a credit card statement',
    'things in a tax return',
    'things in a legal brief',
    'things in a courtroom transcript',
    'things in a school cafeteria',
    'things in a principal office',
    'things in a science classroom',
    'things in a college orientation',
    'things in a campus bookstore',
    'things in a library circulation desk',
    'things in a makerspace',
    'things in a city park',
    'things in a community garden',
    'things in a dog park',
    'things in a botanical garden',
    'things in a beach boardwalk',
    'things in a ski rental shop',
    'things in a campground office',
]

# --- Cell 10 ---
# ============================================================
# CELL 5 — Blue Group (LLM Niche Category)
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


def _blue_connection_from_topic(topic: str) -> str:
    t = (topic or '').strip()
    if not t:
        return t
    return t[0].upper() + t[1:] if len(t) > 1 else t.upper()


def _blue_separate_title_llm() -> bool:
    v = (os.environ.get('BURAK_BLUE_SEPARATE_TITLE_LLM') or '').strip().lower()
    return v in ('1', 'true', 'yes', 'on')


def label_blue_group(group: list) -> str:
    banned = ', '.join(group)
    prompt = (
        f'Here are four words from one category in a NYT Connections puzzle '
        f'(BLUE difficulty — thoughtful, slightly lateral):\n'
        f'{", ".join(group)}\n\n'
        f'Write ONE puzzle title (max 6 words) naming what they ALL share.\n\n'
        f'Rules:\n'
        f'- Sound like a real Connections editor: concrete and specific.\n'
        f'- Do NOT include any of these words in the title text: {banned}\n'
        f'- Return only the title, plain text, no quotation marks wrapping the whole line.'
    )
    try:
        resp = claude_client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=48,
            messages=[{'role': 'user', 'content': prompt}],
        )
        raw = resp.content[0].text.strip()
        title = raw.splitlines()[0].strip().strip('"').strip("'").strip()
        return title
    except Exception:
        return 'Related items'


def _blue_title_format_ok(title: str) -> bool:
    t = (title or '').strip()
    if not t:
        return False
    if t.count('"') % 2 == 1:
        return False
    words = [w for w in t.split() if w]
    if len(words) < 2 or len(words) > 12:
        return False
    bad_markers = ('wait,', 'let me', 'reconsider', "doesn't work", '->', '\n')
    low = t.lower()
    if any(m in low for m in bad_markers):
        return False
    return True


def _repair_blue_title(title: str, group: list) -> str:
    t = (title or '').strip()
    if not t:
        return t
    low = t.lower()
    if 'you can get a "' in low or 'you can get an "' in low:
        m = re.search(r'"([^"]+)"', t)
        if m:
            token = m.group(1).strip().upper()
            if token:
                return f'Words that follow "{token}"'
    m = re.search(r'"([^"]+)"', t)
    if m:
        token = m.group(1).strip()
        if token:
            token_u = token.upper()
            if not _blue_title_format_ok(t):
                return f'Words that follow "{token_u}"'
    return t


def _yellow_label_model() -> str:
    m = (os.environ.get('BURAK_YELLOW_LABEL_MODEL') or '').strip()
    return m or 'claude-sonnet-4-6'


def _yellow_verify_model() -> str:
    m = (os.environ.get('BURAK_YELLOW_VERIFY_MODEL') or '').strip()
    return m or 'claude-sonnet-4-6'


def _yellow_label_max_tokens() -> int:
    try:
        return max(32, min(128, int(os.environ.get('BURAK_YELLOW_LABEL_MAX_TOKENS', '72'))))
    except Exception:
        return 72


def _yellow_verify_max_tokens() -> int:
    try:
        return max(64, min(200, int(os.environ.get('BURAK_YELLOW_VERIFY_MAX_TOKENS', '110'))))
    except Exception:
        return 110


def verify_yellow_group(group: list, connection: str) -> tuple:
    prompt = (
        f'This is proposed as the YELLOW (easiest) Connections group:\n\n'
        f'Title: {connection}\n'
        f'Words: {", ".join(group)}\n\n'
        f'Yellow is the easiest tier: accept the title if it is a fair, natural NYT-style '
        f'category for casual solvers. Reject only when there is a clear problem.\n\n'
        f'1. Does the title reasonably describe ALL four words for a normal player?\n'
        f'2. Reject if any of these CLEAR errors exist:\n'
        f'   - A word from the group appears in the title itself\n'
        f'   - The title uses "follow X" or "precede X" but not every word works\n'
        f'   - One word is clearly the odd one out factually\n'
        f'   - Nationality or demonym adjectives (GERMAN, JAPANESE, EUROPEAN, …) are cast as '
        f'"types of" animals, breeds, jobs, or things they are not — invalid\n'
        f'   - Mixing proper names, brands, or abbreviations with ordinary words unless the title '
        f'genuinely names one tight real-world set\n'
        f'   - Over-broad "can be / could be + adjective" categories (e.g. "Things that can be blue") '
        f'where the trait applies loosely to countless nouns, or one word is an action/abstract noun '
        f'(FLOAT) stretched to fit a "things" list\n'
        f'   - The category is only a loose physical or color association instead of a named, '
        f'tight real-world class all four words belong to\n\n'
        f'Broad umbrella labels are allowed when they still feel natural and useful.\n'
        f'If none of these clear errors apply, return true. Do not reject for picky edge cases.\n\n'
        f'Return only JSON:\n'
        f'{{"valid": true, "reason": "one sentence"}}'
    )
    try:
        resp = claude_client.messages.create(
            model=_yellow_verify_model(),
            max_tokens=_yellow_verify_max_tokens(),
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
            f'Category theme: {topic}\n\n'
            f'Generate exactly 4 single English words that clearly fit that theme.\n\n'
            f'Rules:\n'
            f'- Do NOT use: {avoid_words}\n'
            f'- Do NOT relate to: {avoid_themes}\n'
            f'- Common everyday English — no proper nouns, no multi-word phrases\n'
            f'- Prefer short punchy words\n\n'
            f'Return ONLY this JSON:\n'
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

        if _blue_separate_title_llm():
            connection = None
            verified = False
            for label_try in range(4):
                connection = label_blue_group(group)
                connection = _repair_blue_title(connection, group)
                if not _blue_title_format_ok(connection):
                    print(f'  Title try {label_try + 1}: "{connection}" → rejected (format)')
                    continue
                ok, reason = verify_blue_group(group, connection)
                print(f'  Title try {label_try + 1}: "{connection}" → verified={ok} — {reason}')
                if ok:
                    verified = True
                    break
            if not verified:
                print('  Rejected   : could not verify title')
                continue
        else:
            connection = _blue_connection_from_topic(topic)
            connection = _repair_blue_title(connection, group)
            if not _blue_title_format_ok(connection):
                print(f'  Rejected   : bad title format from topic — "{connection}"')
                continue
            ok, reason = verify_blue_group(group, connection)
            print(f'  Title (from topic): "{connection}" → verified={ok} — {reason}')
            if not ok:
                print('  Rejected   : verifier did not accept words vs topic title')
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
# ============================================================

def find_similar_w2v(anchor, existing_words, topn=200, n=3, *, pos_filter: bool = True):
    al   = anchor.lower()
    excl = set(w.lower() for w in existing_words)
    key  = next((k for k in [al, anchor.upper(), anchor.capitalize()] if k in _get_wv()), None)
    if key is None:
        return []
    ap = None
    if pos_filter:
        try:
            ap = pos_tag([al])[0][1]
        except Exception:
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
        if pos_filter and ap:
            try:
                if pos_tag([word])[0][1][0] != ap[0]:
                    continue
            except Exception:
                pass
        res.append(word)
        if len(res) == n:
            break
    return res


def find_similar_w2v_ranked(
    anchor,
    existing_words,
    topn=280,
    max_cands=12,
    *,
    pos_filter: bool = True,
):
    """
    Same filters as ``find_similar_w2v``, but return up to ``max_cands``
    neighbors in similarity order (for trying alternate 3-word subsets).
    """
    al   = anchor.lower()
    excl = set(w.lower() for w in existing_words)
    key  = next((k for k in [al, anchor.upper(), anchor.capitalize()] if k in _get_wv()), None)
    if key is None:
        return []
    ap = None
    if pos_filter:
        try:
            ap = pos_tag([al])[0][1]
        except Exception:
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
        if pos_filter and ap:
            try:
                if pos_tag([word])[0][1][0] != ap[0]:
                    continue
            except Exception:
                pass
        res.append(word)
        if len(res) >= max_cands:
            break
    return res


def _iter_ranked_w2v_triples(ranked: list, max_trials: int = 14):
    """Yield up to ``max_trials`` distinct (w1,w2,w3) from ranked neighbor lists."""
    n = min(len(ranked), 9)
    if n < 3:
        return
    t = 0
    for combo in combinations(range(n), 3):
        yield [ranked[i] for i in combo]
        t += 1
        if t >= max_trials:
            break


def _imp2_can_build_yellow_group(seed: str, excluded: set) -> bool:
    """
    True iff some triple of w2v neighbors (same filters as yellow, no POS)
    yields a four-word group passing conjugation checks. Uses ranked neighbors
    so this stays aligned with ``generate_yellow_group`` impostor path.
    """
    sl = seed.lower().strip()
    if not sl or sl in excluded:
        return False
    ranked = find_similar_w2v_ranked(
        sl, list(excluded), topn=280, max_cands=12, pos_filter=False
    )
    if len(ranked) < 3:
        return False
    for triple in _iter_ranked_w2v_triples(ranked, max_trials=10):
        group = [seed.upper()] + [w.upper() for w in triple]
        if not group_has_conjugation_issues(group):
            return True
    return False


def label_yellow_group(group: list) -> str:
    words_joined = ', '.join(group)
    banned_in_label = ', '.join(group)
    prompt = (
        f'These 4 words are the YELLOW (easiest) group in NYT Connections: '
        f'{words_joined}.\n\n'
        f'Write ONE connection title (max 6 words) — a NYT-style category name '
        f'a casual solver would find satisfying.\n\n'
        f'Rules:\n'
        f'- The title must fit all four words naturally for a casual solver.\n'
        f'- Slightly broader umbrella categories are okay when they are clear and useful.\n'
        f'- Prefer a **named** class: "Types of ___", "Things in a ___", '
        f'"Ways to ___", "Parts of a ___", "___ in sports", etc.\n'
        f'- Do NOT include any of these words in the title: {banned_in_label}\n'
        f'- Do NOT use vague "Words associated with" / "Things related to".\n'
        f'- NEVER write patterns like "Things that can be [color/adjective]" '
        f'(e.g. "Things that can be blue") — too loose; reject that idea entirely.\n'
        f'- Avoid any title where the link is only "could share one loose physical trait" '
        f'(color, size, hot/cold) unless the title names one obvious tight real-world set.\n'
        f'- Do NOT use nationalities/demonyms as fake parallel "types" (e.g. breeds) '
        f'when the words are just place-derived adjectives.\n'
        f'- If one word is broader than others, this is acceptable only when the category '
        f'still reads as one coherent everyday set.\n'
        f'- Do NOT use "can follow / precede / after / before" framing.\n\n'
        f'If no very tight title fits, output the best clear umbrella category.\n\n'
        f'Return only the title, plain text.'
    )
    try:
        resp = claude_client.messages.create(
            model=_yellow_label_model(),
            max_tokens=_yellow_label_max_tokens(),
            messages=[{'role': 'user', 'content': prompt}]
        )
        raw = resp.content[0].text.strip()
        title = raw.splitlines()[0].strip()
        title = title.strip('"').strip("'").strip()
        return title
    except Exception:
        return f'Words related to {group[0]}'


def _is_bad_yellow_title_format(title: str, group: list) -> tuple[bool, str]:
    t = (title or '').strip()
    if not t:
        return True, 'empty title'
    lower = t.lower()
    words = [w for w in t.replace('"', '').split() if w]
    if len(words) < 2 or len(words) > 7:
        return True, 'title length out of range (2-7 words)'
    bad_markers = ('wait,', 'let me', 'reconsider', "doesn't work", '->', '—', '\n')
    if any(m in lower for m in bad_markers):
        return True, 'rambling/meta text detected'
    if t.count('"') % 2 == 1:
        return True, 'unbalanced quotes'
    if lower.startswith('words that follow') or lower.startswith('words that precede'):
        return True, 'yellow should not use follow/precede framing'
    banned_fragments = (
        ' can follow ',
        ' can precede ',
        ' that follow ',
        ' that precede ',
    )
    if any(f in lower for f in banned_fragments):
        return True, 'yellow should not use follow/precede/after/before framing'
    # Vague "can be + adjective" yellows (e.g. "Things that can be blue")
    if 'things that can be ' in lower or 'stuff that can be ' in lower:
        return True, 'too-vague "things that can be …" category'
    if 'things that could be ' in lower or 'stuff that could be ' in lower:
        return True, 'too-vague "things that could be …" category'
    group_lower = {w.lower() for w in group}
    title_tokens = {
        tok.strip('".,!?;:()[]{}').lower()
        for tok in t.split()
        if tok.strip('".,!?;:()[]{}')
    }
    if group_lower & title_tokens:
        return True, 'title includes group word(s)'
    return False, ''


def _repair_yellow_title(title: str) -> str:
    """Fix unbalanced quotes from LLM output."""
    t = (title or '').strip()
    if t.count('"') % 2 == 1:
        t = t.replace('"', '')
    return t


def _yellow_title_try_budget() -> int:
    """Label + verify cycles per word2vec group (Sonnet label; cap cost via env)."""
    try:
        return max(1, min(4, int(os.environ.get('BURAK_YELLOW_TITLE_MAX_TRIES', '1'))))
    except Exception:
        return 1


def _yellow_imp2_triple_budget() -> int:
    """How many imp2 neighbor triples to evaluate before falling back."""
    try:
        return max(1, min(12, int(os.environ.get('BURAK_YELLOW_IMP2_TRIPLE_MAX_TRIES', '4'))))
    except Exception:
        return 4


def yellow_connection_with_retries(group: list, max_tries: int | None = None):
    if max_tries is None:
        max_tries = _yellow_title_try_budget()
    for attempt in range(max_tries):
        label = label_yellow_group(group)
        label = _repair_yellow_title(label)
        bad, why = _is_bad_yellow_title_format(label, group)
        if bad:
            print(f'    yellow title try {attempt + 1}: "{label}" → False — {why}')
            continue
        ok, reason = verify_yellow_group(group, label)
        print(f'    yellow title try {attempt + 1}: "{label}" → {ok} — {reason}')
        if ok:
            return label
    return None


def _yellow_seed_budget() -> int:
    try:
        return max(6, min(40, int(os.environ.get('BURAK_YELLOW_SEED_MAX_ATTEMPTS', '20'))))
    except Exception:
        return 20


def generate_yellow_group(existing_words, impostor=None, max_attempts=None):
    if max_attempts is None:
        max_attempts = _yellow_seed_budget()
    excl  = set(w.lower() for w in existing_words)
    tried = set()

    if impostor:
        impostor_lower = impostor.lower()
        excl_with_imp = excl | {impostor_lower}
        ranked = find_similar_w2v_ranked(
            impostor_lower,
            list(excl_with_imp) + list(tried),
            topn=320,
            max_cands=12,
            pos_filter=False,
        )
        for triple in _iter_ranked_w2v_triples(
            ranked, max_trials=_yellow_imp2_triple_budget()
        ):
            group = [impostor.upper()] + [w.upper() for w in triple]
            if group_has_conjugation_issues(group):
                continue
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

    Impostor logic (staged):
      1. For each purple word as anchor (shuffled), pick ``imp1`` that can anchor green.
      2. Build green around ``imp1``.
      3. Only then pick ``imp2`` from w2vec neighbors of the anchor, excluding
         purple ∪ green so yellow feasibility matches the real board.
      4. ``imp2`` seeds yellow (word2vec).

    If no anchor yields a full (imp1 green + imp2 yellow) bundle, green is
    generated without impostors. A correct puzzle beats a forced mechanic.
    """
    used: list[str] = []

    # Purple
    purple = generate_purple_group(n_candidates=8)
    if not purple:
        return None
    used += purple['group']

    impostor_result: dict | None = None
    green: dict | None = None

    anchors = list(purple['group'])
    random.shuffle(anchors)

    for anchor_word in anchors:
        print(f'  Staged impostor — trying anchor: {anchor_word}')

        imp1, imp1_mechanism = find_impostor_green(
            purple_word=anchor_word,
            purple_group=purple['group'],
            exclude_extra=[],
        )
        if imp1 is None:
            print('    No green-capable imp1 found')
            continue

        print(f'    imp1 (green anchor): {imp1.upper()} [{imp1_mechanism}]')

        partial = {
            'anchor':         anchor_word.upper(),
            'imp1':           imp1.upper(),
            'imp1_mechanism': imp1_mechanism,
            'imp1_target':    'green',
            'imp2_target':    'yellow',
        }

        green_try = generate_rhyme_group(
            existing_words=used,
            impostor=imp1,
            impostor_result=partial,
        )
        if not green_try:
            print('    Green generation failed for this anchor')
            continue

        if not green_try.get('has_impostor'):
            print('    Green did not retain impostor — trying next anchor')
            continue

        imp2, _ = find_impostor_w2v_seed(
            purple_word=anchor_word,
            purple_group=purple['group'],
            exclude_extra=green_try['group'],
        )
        if imp2 is None:
            print(
                '    No w2v-capable imp2 after green '
                '(neighbors blocked by purple ∪ green) — trying next anchor'
            )
            continue

        print(f'    imp2 (yellow seed): {imp2.upper()}')
        fake_label = label_fake_connection(
            partial['anchor'], partial['imp1'], imp2.upper()
        )
        print(f'    Fake connection: "{fake_label}"')

        impostor_result = {
            **partial,
            'imp2':            imp2.upper(),
            'fake_connection': fake_label,
        }
        green = green_try
        break

    if impostor_result is None:
        print('  No staged impostor bundle worked — generating green without impostor')
        green = generate_rhyme_group(
            existing_words=used,
            impostor=None,
            impostor_result=None,
        )
        if not green:
            return None

    imp_for_green  = impostor_result['imp1'] if impostor_result else None
    imp_for_yellow = impostor_result['imp2'] if impostor_result else None

    used += green['group']

    # Yellow (with imp2 as seed) — built before blue so imp2 has a larger word pool
    yellow = generate_yellow_group(existing_words=used, impostor=imp_for_yellow)
    if not yellow:
        return None
    used += yellow['group']

    # Verify imp2 actually landed in yellow.
    # Must check before blue is generated so we can clear cleanly.
    if impostor_result and imp_for_yellow:
        if impostor_result['imp2'] not in yellow['group']:
            print(
                f'  imp2 "{impostor_result["imp2"]}" not in yellow group — '
                f'clearing impostor_result to avoid broken fake connection'
            )
            impostor_result = None
            green['has_impostor'] = False
            green.pop('impostor', None)
            green.pop('anchor', None)
            green.pop('fake_connection', None)

    # Blue — generated last, least constrained by impostor system
    blue = generate_blue_group(
        existing_words=used,
        purple_conn=purple.get('connection', ''),
        green_conn=green.get('connection', '')
    )
    if not blue:
        return None
    used += blue['group']

    # Board diversity check — reject if green+purple words are all too long
    # This prevents boards where 8+ words are long corporate/academic terms
    # that all look the same to a player scanning the grid
    green_purple_words = green['group'] + purple['group']
    avg_len = sum(len(w) for w in green_purple_words) / len(green_purple_words)
    if avg_len > 8.0:
        print(
            f'  Rejecting puzzle: green+purple avg word length {avg_len:.1f} > 8.0 '
            f'(board looks visually uniform — retrying)'
        )
        return None

    # Also reject if more than 5 of the 16 board words exceed 10 characters
    all_16 = yellow['group'] + green['group'] + blue['group'] + purple['group']
    long_words = [w for w in all_16 if len(w) > 10]
    if len(long_words) > 5:
        print(
            f'  Rejecting puzzle: {len(long_words)} words exceed 10 chars {long_words} '
            f'(board looks visually uniform — retrying)'
        )
        return None

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