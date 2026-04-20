# ============================================================
# adreama_module.py
# Auto-extracted from adreama_connections_ai.ipynb
# Entry point: generate_one_puzzle_entry(verbose=False)
# ============================================================

import os
# API key must be set as env var OPENAI_API_KEY before importing.
# Uses the public OpenAI API (api.openai.com); optional OPENAI_CHAT_MODEL overrides the model id.

import os, json, csv, time, random, re
from typing import Optional
from openai import OpenAI
from collections import Counter

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
MODEL = os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o')
MAX_RETRIES = 4
RETRY_DELAY = 0.8

# Files that persist across sessions
PUZZLES_CSV       = 'puzzles.csv'
USED_WORDS_FILE   = 'used_words.json'
USED_CATS_FILE    = 'used_categories.json'
USED_BOARDS_FILE  = 'used_boards.json'

def call_api(prompt: str, retries: int = MAX_RETRIES) -> Optional[dict]:
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL, max_tokens=1400, temperature=1.0,
                messages=[{'role': 'user', 'content': prompt}]
            )
            text = r.choices[0].message.content.strip()
            if text.startswith('```'):
                text = text.split('```')[1]
                if text.startswith('json'): text = text[4:]
            return json.loads(text.strip())
        except json.JSONDecodeError:
            print(f'  JSON parse failed ({attempt+1}/{retries}), retrying...')
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f'  API error ({attempt+1}/{retries}): {e}')
            time.sleep(RETRY_DELAY)
    return None

print('Config loaded.')

def load_json_set(path: str) -> set:
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()

def save_json_set(path: str, s: set) -> None:
    with open(path, 'w') as f:
        json.dump(sorted(s), f, indent=2)

# Load everything from disk
USED_WORDS        = load_json_set(USED_WORDS_FILE)
USED_CATS         = load_json_set(USED_CATS_FILE)
USED_BOARDS       = load_json_set(USED_BOARDS_FILE)
USED_PURPLE_TYPES = load_json_set('used_purple_types.json')

def save_memory():
    save_json_set(USED_WORDS_FILE,          USED_WORDS)
    save_json_set(USED_CATS_FILE,           USED_CATS)
    save_json_set(USED_BOARDS_FILE,         USED_BOARDS)
    save_json_set('used_purple_types.json', USED_PURPLE_TYPES)
    save_json_set('used_yellow_cats.json',  USED_YELLOW_CATS)

def board_signature(puzzle: dict) -> str:
    words = sorted(
        w.lower().strip()
        for g in puzzle['groups'].values()
        for w in g['words']
    )
    return '|'.join(words)

def register_puzzle(puzzle):
    for colour, g in puzzle['groups'].items():
        USED_WORDS.update(w.lower() for w in g['words'])
    ycat = puzzle['groups']['yellow']['category'].lower().strip()
    USED_YELLOW_CATS.add(ycat)
    pcat = puzzle['groups']['purple']['category'].lower().strip()
    USED_PURPLE_TYPES.add(get_purple_type(pcat))
    sig = board_signature(puzzle)
    USED_BOARDS.add(sig)
    save_memory()

def reset_memory():
    global USED_WORDS, USED_CATS, USED_BOARDS, USED_PURPLE_TYPES, USED_YELLOW_CATS
    USED_WORDS        = set()
    USED_CATS         = set()
    USED_BOARDS       = set()
    USED_PURPLE_TYPES = set()
    USED_YELLOW_CATS  = set()
    save_memory()
    print('Memory cleared.')

puzzles_so_far = 0
if os.path.exists(PUZZLES_CSV):
    with open(PUZZLES_CSV) as f:
        puzzles_so_far = sum(1 for row in csv.reader(f)) - 1
    puzzles_so_far = puzzles_so_far // 4

print(f'Memory loaded.')
print(f'  Words used   : {len(USED_WORDS)}')
print(f'  Categories   : {len(USED_CATS)}')
print(f'  Boards saved : {len(USED_BOARDS)}')
print(f'  Puzzles in CSV: {puzzles_so_far}')

# Each web is a list of 4 (colour, cat_a, cat_b) tuples.
# cat_a = the real category the word will be placed in
# cat_b = the trap category players will mistake it for
# Shared categories across slots create the false group pressure.

CATEGORY_WEBS = [
    # birds / speed / door parts / seasons
    [('yellow','birds','words meaning fast'),
     ('green', 'words meaning fast','parts of a door'),
     ('blue',  'birds','construction equipment'),
     ('purple','parts of a door','seasons')],

    # dances / movement / music / boxing
    [('yellow','dances','verbs of movement'),
     ('green', 'verbs of movement','boxing terms'),
     ('blue',  'dances','music genres'),
     ('purple','boxing terms','cooking methods')],

    # hairstyles / fish / tools / cutting
    [('yellow','hairstyles','types of fish'),
     ('green', 'types of fish','kitchen tools'),
     ('blue',  'hairstyles','verbs meaning to cut'),
     ('purple','kitchen tools','verbs meaning to cut')],

    # card games / cheating / clothing / crime
    [('yellow','card game terms','verbs meaning to cheat'),
     ('green', 'verbs meaning to cheat','crime terms'),
     ('blue',  'card game terms','clothing items'),
     ('purple','clothing items','crime terms')],

    # cooking / chemistry / collapse / sounds
    [('yellow','cooking techniques','chemistry processes'),
     ('green', 'chemistry processes','sounds'),
     ('blue',  'cooking techniques','verbs meaning to collapse'),
     ('purple','sounds','verbs meaning to collapse')],

    # weapons / striking / music / body parts
    [('yellow','weapons','verbs meaning to strike'),
     ('green', 'verbs meaning to strike','music terms'),
     ('blue',  'weapons','body parts'),
     ('purple','music terms','body parts')],

    # legal terms / defence / architecture / games
    [('yellow','legal terms','verbs meaning to defend'),
     ('green', 'verbs meaning to defend','architecture terms'),
     ('blue',  'legal terms','board game terms'),
     ('purple','architecture terms','board game terms')],

    # theatre / performance / royalty / cards
    [('yellow','theatre terms','verbs meaning to perform'),
     ('green', 'verbs meaning to perform','royalty terms'),
     ('blue',  'theatre terms','card game terms'),
     ('purple','royalty terms','card game terms')],

    # photography / exposure / clothing / light
    [('yellow','photography terms','verbs meaning to expose'),
     ('green', 'verbs meaning to expose','clothing items'),
     ('blue',  'photography terms','types of light'),
     ('purple','clothing items','types of light')],

    # cricket / hitting / food / shapes
    [('yellow','cricket terms','verbs meaning to hit'),
     ('green', 'verbs meaning to hit','food items'),
     ('blue',  'cricket terms','shapes'),
     ('purple','food items','shapes')],

    # horses / running / fabrics / colours
    [('yellow','horse-racing terms','verbs meaning to run'),
     ('green', 'verbs meaning to run','fabric types'),
     ('blue',  'horse-racing terms','colours'),
     ('purple','fabric types','colours')],

    # printing / pressing / crime / fonts
    [('yellow','printing terms','verbs meaning to press'),
     ('green', 'verbs meaning to press','crime terms'),
     ('blue',  'printing terms','font styles'),
     ('purple','crime terms','font styles')],

    # animals / collective nouns / politics / crowding
    [('yellow','animals','collective nouns for groups'),
     ('green', 'collective nouns for groups','political terms'),
     ('blue',  'animals','verbs meaning to crowd'),
     ('purple','political terms','verbs meaning to crowd')],

    # boxing / sideways movement / dance / emotions
    [('yellow','boxing terms','verbs meaning to move sideways'),
     ('green', 'verbs meaning to move sideways','dance moves'),
     ('blue',  'boxing terms','emotions'),
     ('purple','dance moves','emotions')],

    # medicine / cutting / carpentry / sharp sounds
    [('yellow','medical terms','verbs meaning to cut'),
     ('green', 'verbs meaning to cut','carpentry tools'),
     ('blue',  'medical terms','sharp sounds'),
     ('purple','carpentry tools','sharp sounds')],

    # geology / layers / music / shades
    [('yellow','geology terms','verbs meaning to form layers'),
     ('green', 'verbs meaning to form layers','music terms'),
     ('blue',  'geology terms','shades of colour'),
     ('purple','music terms','shades of colour')],

    # trees / growth / money / emotions
    [('yellow','types of tree','verbs meaning to grow'),
     ('green', 'verbs meaning to grow','money terms'),
     ('blue',  'types of tree','emotions'),
     ('purple','money terms','emotions')],

    # sailing / tying / fabric / knots
    [('yellow','sailing terms','verbs meaning to tie'),
     ('green', 'verbs meaning to tie','fabric types'),
     ('blue',  'sailing terms','types of knot'),
     ('purple','fabric types','types of knot')],

    # dogs / loyalty / jobs / sport positions
    [('yellow','dog breeds','adjectives meaning loyal or faithful'),
     ('green', 'adjectives meaning loyal or faithful','job titles'),
     ('blue',  'dog breeds','sports positions'),
     ('purple','job titles','sports positions')],
]



MORE_CATEGORY_WEBS = [

# sports / scoring / food / shapes
[('yellow','sports terms','verbs meaning to score'),
 ('green','verbs meaning to score','food items'),
 ('blue','sports terms','geometric shapes'),
 ('purple','food items','geometric shapes')],

# driving / movement / finance / speed
[('yellow','driving terms','verbs meaning to move'),
 ('green','verbs meaning to move','finance terms'),
 ('blue','driving terms','words meaning fast'),
 ('purple','finance terms','words meaning fast')],

# cooking / heating / science / reactions
[('yellow','cooking techniques','verbs meaning to heat'),
 ('green','verbs meaning to heat','science terms'),
 ('blue','cooking techniques','chemical reactions'),
 ('purple','science terms','chemical reactions')],

# music / playing / games / roles
[('yellow','music terms','verbs meaning to play'),
 ('green','verbs meaning to play','game roles'),
 ('blue','music terms','board game terms'),
 ('purple','game roles','board game terms')],

# fashion / wearing / seasons / weather
[('yellow','clothing items','verbs meaning to wear'),
 ('green','verbs meaning to wear','seasonal terms'),
 ('blue','clothing items','weather terms'),
 ('purple','seasonal terms','weather terms')],

# construction / building / geometry / tools
[('yellow','construction terms','verbs meaning to build'),
 ('green','verbs meaning to build','geometry terms'),
 ('blue','construction terms','hand tools'),
 ('purple','geometry terms','hand tools')],

# animals / movement / verbs / emotions
[('yellow','animals','verbs meaning to move quickly'),
 ('green','verbs meaning to move quickly','emotions'),
 ('blue','animals','verbs meaning to act aggressively'),
 ('purple','emotions','verbs meaning to act aggressively')],

# school / learning / writing / grammar
[('yellow','school terms','verbs meaning to learn'),
 ('green','verbs meaning to learn','writing terms'),
 ('blue','school terms','grammar terms'),
 ('purple','writing terms','grammar terms')],

# ocean / water / motion / travel
[('yellow','ocean terms','verbs meaning to flow'),
 ('green','verbs meaning to flow','travel terms'),
 ('blue','ocean terms','verbs meaning to move'),
 ('purple','travel terms','verbs meaning to move')],

# photography / light / exposure / vision
[('yellow','photography terms','verbs meaning to expose'),
 ('green','verbs meaning to expose','vision terms'),
 ('blue','photography terms','light sources'),
 ('purple','vision terms','light sources')],

# crime / law / stealing / money
[('yellow','crime terms','verbs meaning to steal'),
 ('green','verbs meaning to steal','money terms'),
 ('blue','crime terms','legal terms'),
 ('purple','money terms','legal terms')],

# farming / growth / nature / seasons
[('yellow','farming terms','verbs meaning to grow'),
 ('green','verbs meaning to grow','nature terms'),
 ('blue','farming terms','seasonal cycles'),
 ('purple','nature terms','seasonal cycles')],

# art / drawing / shapes / design
[('yellow','art terms','verbs meaning to draw'),
 ('green','verbs meaning to draw','geometric shapes'),
 ('blue','art terms','design elements'),
 ('purple','geometric shapes','design elements')],

# travel / movement / directions / maps
[('yellow','travel terms','verbs meaning to move'),
 ('green','verbs meaning to move','directional terms'),
 ('blue','travel terms','map features'),
 ('purple','directional terms','map features')],

# medicine / healing / body / injury
[('yellow','medical terms','verbs meaning to heal'),
 ('green','verbs meaning to heal','body parts'),
 ('blue','medical terms','injury terms'),
 ('purple','body parts','injury terms')],

# retail / buying / selling / finance
[('yellow','retail terms','verbs meaning to buy'),
 ('green','verbs meaning to buy','finance terms'),
 ('blue','retail terms','verbs meaning to sell'),
 ('purple','finance terms','verbs meaning to sell')],

# military / attack / defence / strategy
[('yellow','military terms','verbs meaning to attack'),
 ('green','verbs meaning to attack','defence terms'),
 ('blue','military terms','strategy terms'),
 ('purple','defence terms','strategy terms')],

# weather / wind / motion / sailing
[('yellow','weather terms','verbs meaning to blow'),
 ('green','verbs meaning to blow','sailing terms'),
 ('blue','weather terms','verbs meaning to move'),
 ('purple','sailing terms','verbs meaning to move')],

# tech / computing / data / storage
[('yellow','computer terms','verbs meaning to store'),
 ('green','verbs meaning to store','data terms'),
 ('blue','computer terms','file types'),
 ('purple','data terms','file types')],

# games / scoring / competition / ranking
[('yellow','game terms','verbs meaning to score'),
 ('green','verbs meaning to score','competition terms'),
 ('blue','game terms','ranking terms'),
 ('purple','competition terms','ranking terms')],

# language / speaking / writing / grammar
[('yellow','language terms','verbs meaning to speak'),
 ('green','verbs meaning to speak','writing terms'),
 ('blue','language terms','grammar terms'),
 ('purple','writing terms','grammar terms')],

# finance / saving / investing / risk
[('yellow','finance terms','verbs meaning to save'),
 ('green','verbs meaning to save','investment terms'),
 ('blue','finance terms','risk terms'),
 ('purple','investment terms','risk terms')],

# architecture / building / rooms / design
[('yellow','architecture terms','verbs meaning to build'),
 ('green','verbs meaning to build','room types'),
 ('blue','architecture terms','design styles'),
 ('purple','room types','design styles')],

# music / sound / rhythm / tempo
[('yellow','music terms','verbs meaning to sound'),
 ('green','verbs meaning to sound','rhythm terms'),
 ('blue','music terms','tempo markings'),
 ('purple','rhythm terms','tempo markings')],

# science / testing / lab / measurement
[('yellow','science terms','verbs meaning to test'),
 ('green','verbs meaning to test','lab equipment'),
 ('blue','science terms','measurement units'),
 ('purple','lab equipment','measurement units')],

# food / eating / cooking / taste
[('yellow','food terms','verbs meaning to eat'),
 ('green','verbs meaning to eat','cooking methods'),
 ('blue','food terms','taste descriptors'),
 ('purple','cooking methods','taste descriptors')],

# transport / driving / roads / signals
[('yellow','transport terms','verbs meaning to drive'),
 ('green','verbs meaning to drive','road terms'),
 ('blue','transport terms','traffic signals'),
 ('purple','road terms','traffic signals')],

# emotions / feeling / behavior / reactions
[('yellow','emotions','verbs meaning to feel'),
 ('green','verbs meaning to feel','behavior terms'),
 ('blue','emotions','reaction terms'),
 ('purple','behavior terms','reaction terms')],

# sports / training / fitness / body
[('yellow','fitness terms','verbs meaning to train'),
 ('green','verbs meaning to train','body parts'),
 ('blue','fitness terms','exercise types'),
 ('purple','body parts','exercise types')],
]

CATEGORY_WEBS.extend(MORE_CATEGORY_WEBS)
print("Total webs:", len(CATEGORY_WEBS))

# Track which webs have been used recently to avoid repetition
_web_usage = {i: 0 for i in range(len(CATEGORY_WEBS))}

def pick_web() -> tuple[int, list]:
    """Pick the least-recently-used web."""
    idx = min(_web_usage, key=_web_usage.get)
    _web_usage[idx] += 1
    return idx, CATEGORY_WEBS[idx]

print(f'{len(CATEGORY_WEBS)} category webs loaded.')

import unicodedata

# ── Constants ─────────────────────────────────────────────────────────────────

PROMPT = '''
You are building an NYT Connections puzzle.

STEP 1 — BUILD PURPLE (hardest group)
Purple connections must be real and verifiable — not invented.
If you are not certain all 4 words genuinely belong to the category,
choose a different category. Never fabricate a connection.
Purple must have TWO properties simultaneously:

  A) The connection is SUBTLE — not obvious even after you see it
     Good types:
     - collective nouns for unusual professions or specific animals
     - words that complete a non-obvious hidden phrase
     - cryptic category where membership is genuinely surprising
     - unexpected membership where the category is narrow and specific
     - words associated with a hidden theme where each word has a
       strong alternate meaning that points elsewhere

     NOT acceptable:
     - collective nouns for common animals — banned
     - contronyms or heteronyms — too mechanical
     - chess pieces, card suits, planets, dances — too obvious
     - any category containing: multiple meanings, words that are,
       words that can be, dual, also, both, function as

  B) Exactly ONE word is the trap word
     One word in purple strongly suggests a completely different
     simpler category. Players will be certain it belongs elsewhere.
     The other 3 purple words are clean unambiguous members.

     The trap word's fake category must be something DIFFERENT from
     any example you have seen before. Do not reuse seating, sounds,
     tools, fruit, or any other overused trap category.
     
Only use collective nouns you are 100% certain are real. If in doubt, choose a different purple type entirely

STEP 2 — BUILD YELLOW, GREEN, BLUE

Yellow naturally contains words from the trap word's fake category,
creating a false trail without forcing anything.

YELLOW = easiest. Immediately obvious. Pure membership.
  BANNED yellow categories — never use these:
  seasons, days, months, chess pieces, musical notes, school supplies,
  cardinal directions, types of fruit, kitchen appliances,
  kitchen utensils, types of bread, cooking methods, types of seating,
  things that can be cracked, body parts, colors, simple animals,
  common verbs, board games, types of pasta, types of cake,
  {banned_yellow_cats}

GREEN = medium. One inferential step required.
  The connection must not be instantly obvious from the words alone.
  Good: oblique categories, hidden themes, famous people sharing
        a first name, things associated with an unexpected concept
  Bad: plain semantic lists where membership is immediate

BLUE = hard. Linguistic or structural only.
  Must be one of: phrase completion, compound words, domain vocabulary,
  informal words for the same concept, words following or preceding
  a hidden word.
  Never just a semantic category.

CRITICAL DIVERSITY RULE:
Every puzzle must use completely different categories for all four
groups. Do not reuse any category, theme, or word set from any
previous puzzle. If your first instinct is something familiar,
discard it and find a completely different angle.

LABEL RULES:
  Yellow: plain and specific
  Green: slightly oblique
  Blue: shows the structural pattern explicitly
  Purple: names WHAT the words are, never HOW the trick works
    GOOD: "collective nouns for professions", "characters in Peter Pan",
          "___ roll", "things that can go flat"
    BAD:  "words that are both X and Y", "words with multiple meanings"
    BANNED label words: also, both, dual, function as, used as,
                        multiple meanings, polysemous, words that are,
                        words that can be

WORD RULES:
  Exactly 16 distinct entries across all 4 groups.
  Single or multi-word entries, maximum 3 words.
  Lowercase, no hyphens — use spaces instead.
  No obscure words — all known to a general audience.
  No two entries where one derives from the other
  (e.g. hazel / hazelnut, head / headline — banned).
  Do NOT use: {banned_words}
  Do NOT repeat these purple types (overused): {banned_purple_types}
  Last purple category was: {last_purple_cat} — do not repeat it.

Return JSON only — no explanation, no markdown:
{{
  "purple":            {{"category": "...", "words": ["w1","w2","w3","w4"]}},
  "trap_word":         "...",
  "trap_impersonates": "...",
  "yellow":            {{"category": "...", "words": ["w1","w2","w3","w4"]}},
  "green":             {{"category": "...", "words": ["w1","w2","w3","w4"]}},
  "blue":              {{"category": "...", "words": ["w1","w2","w3","w4"]}}
}}
'''

BANNED_CATEGORY_SNIPPETS = [
    'noun and verb',
    'nouns and verbs',
    'multiple meanings',
    'can mean both',
    'derived from surnames',
    'brand names',
]

BANNED_PURPLE_SIGNALS = [
    'words that are both',
    'words with multiple meanings',
    'multiple meanings',
    'polysemous',
    'dual meaning',
    'words that function',
    'words that serve',
]

BANNED_YELLOW_CATS = {
    'types of hat', 'hats', 'types of cheese', 'types of bread',
    'colors', 'colours', 'school supplies', 'musical instruments',
    'instruments', 'cardinal directions', 'compass points', 'seasons',
    'days of the week', 'months of the year', 'musical notes',
    'chemical elements', 'blood cells', 'types of seating',
    'things that can be cracked', 'types of fruit', 'kitchen appliances',
    'kitchen utensils', 'cooking methods', 'body parts', 'simple animals',
    'board games', 'types of pasta', 'types of cake',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalise_entry(w: str) -> str:
    w = unicodedata.normalize('NFKD', w)
    w = w.encode('ascii', 'ignore').decode('ascii')
    w = re.sub(r'[^a-z ]', '', w.lower().strip())
    w = re.sub(r' +', ' ', w).strip()
    return w

def validate_category_quality(groups):
    for colour, g in groups.items():
        cat = g['category'].lower().strip()
        if any(s in cat for s in BANNED_CATEGORY_SNIPPETS):
            return False, f'{colour} category too meta or broad: {cat}'
        if 'line' in cat and any(w.endswith('line') for w in g['words']):
            return False, f'{colour} uses completed -line compounds'
    return True, ''

def parse_result(result: dict) -> Optional[dict]:
    if result is None:
        return None
    try:
        groups = {
            'yellow': result['yellow'],
            'green':  result['green'],
            'blue':   result['blue'],
            'purple': result['purple'],
        }
        for g in groups.values():
            g['words'] = [normalise_entry(w) for w in g['words']]
        return {
            'groups':            groups,
            'trap_word':         normalise_entry(result.get('trap_word', '')),
            'trap_impersonates': result.get('trap_impersonates', ''),
        }
    except (KeyError, TypeError) as e:
        print(f'  parse error: {e}')
        return None

def validate_puzzle(result: dict) -> tuple[bool, str]:
    groups = result.get('groups', {})
    if set(groups.keys()) != {'yellow', 'green', 'blue', 'purple'}:
        return False, f'wrong colours: {list(groups.keys())}'
    all_words = []
    for colour, g in groups.items():
        words = g.get('words', [])
        if len(words) != 4:
            return False, f'{colour} has {len(words)} words'
        for w in words:
            w = w.lower().strip()
            if not re.fullmatch(r'[a-z]+( [a-z]+){0,2}', w):
                return False, f'invalid entry: {w}'
            if len(w.split()) > 3:
                return False, f'entry too long: {w}'
        if not g.get('category', '').strip():
            return False, f'{colour} missing category'
        all_words.extend(w.lower().strip() for w in words)
    if len(all_words) != len(set(all_words)):
        dupes = {w for w in all_words if all_words.count(w) > 1}
        return False, f'duplicate words: {dupes}'
    ycat = groups['yellow']['category'].lower().strip()
    if ycat in BANNED_YELLOW_CATS:
        return False, f'yellow too easy: {ycat}'
    pcat = groups['purple']['category'].lower().strip()
    if any(s in pcat for s in BANNED_PURPLE_SIGNALS):
        return False, f'purple label reveals trick: {pcat}'
    ok, reason = validate_category_quality(groups)
    if not ok:
        return False, reason
    return True, ''

def check_freshness(p):
    sig = board_signature(p)
    if sig in USED_BOARDS:
        return False, 'exact board already exists'
    return True, 'OK'

def get_purple_type(category: str) -> str:
    cat = category.lower()
    if 'collective noun' in cat:        return 'collective_nouns'
    if 'contranym' in cat:              return 'contronyms'
    if 'heteronym' in cat:              return 'heteronyms'
    if 'detective' in cat:              return 'famous_detectives'
    if 'kitchen' in cat:                return 'kitchen'
    if 'fruit' in cat:                  return 'fruit'
    if 'character' in cat:              return 'characters'
    if 'hidden' in cat:                 return 'hidden_theme'
    if 'flat' in cat:                   return 'things_go_flat'
    if 'phrase' in cat or '___' in cat: return 'phrase_completion'
    return 'other'

# ── Generator ─────────────────────────────────────────────────────────────────

LAST_PURPLE_CAT  = ''
USED_YELLOW_CATS = load_json_set('used_yellow_cats.json')

def generate_one_puzzle(verbose: bool = True) -> Optional[dict]:
    global LAST_PURPLE_CAT

    recent = list(USED_WORDS)[-40:]
    if len(USED_WORDS) > 40:
        older = random.sample(list(USED_WORDS)[:-40], min(40, len(USED_WORDS) - 40))
    else:
        older = []
    banned_words_str  = ', '.join(sorted(set(recent + older))) or 'none'
    banned_yellow_str = ', '.join(sorted(USED_YELLOW_CATS)) or 'none'

    purple_type_counts = Counter(USED_PURPLE_TYPES)
    overused           = {t for t, n in purple_type_counts.items() if n >= 3}
    banned_purple_str  = ', '.join(overused) or 'none'

    prompt = PROMPT.format(
        banned_words=banned_words_str,
        banned_yellow_cats=banned_yellow_str,
        banned_purple_types=banned_purple_str,
        last_purple_cat=LAST_PURPLE_CAT,
    )

    for attempt in range(MAX_RETRIES):
        if verbose:
            print(f'  Attempt {attempt+1}/{MAX_RETRIES}...', end=' ', flush=True)

        result = call_api(prompt)
        if result is None:
            if verbose: print('API failed')
            continue

        parsed = parse_result(result)
        if parsed is None:
            if verbose: print('parse failed')
            continue

        ok, reason = validate_puzzle(parsed)
        if not ok:
            if verbose: print(f'invalid: {reason}')
            continue

        ok, reason = check_freshness(parsed)
        if not ok:
            if verbose: print(f'not fresh: {reason}')
            continue

        grid = []
        for colour in ['yellow', 'green', 'blue', 'purple']:
            grid.extend(parsed['groups'][colour]['words'])
        random.shuffle(grid)

        parsed['grid'] = grid
        LAST_PURPLE_CAT = parsed['groups']['purple']['category'].lower().strip()

        if verbose: print('OK')
        return parsed

    return None

print('Generator loaded.')


# ============================================================
# Entry point for AdreamaAIAdapter in master_generator
# ============================================================

def generate_one_puzzle_entry(verbose: bool = False):
    """
    Called by AdreamaAIAdapter.generate().
    Returns puzzle dict or None.
    """
    return generate_one_puzzle(verbose=verbose)