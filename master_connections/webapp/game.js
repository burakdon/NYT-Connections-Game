// Static Connections UI (adapted from Kevin's Infinite Connections webapp).
// Loads puzzle data from puzzles.json (see scripts/export_web_puzzles.py).

let allPuzzles = [];
let currentPuzzle = null;

let state = {
    words: [],
    selected: new Set(),
    solvedGroups: [],
    mistakesLeft: 4,
    gameOver: false,
};

// -- Initialization ---------------------------------------------------------

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const resp = await fetch('puzzles.json');
        allPuzzles = await resp.json();
        document.getElementById('puzzle-count').textContent =
            allPuzzles.length + ' puzzles available';
        loadNewPuzzle();
    } catch (e) {
        showMessage('Failed to load puzzles', 'wrong');
    }
});

function loadNewPuzzle() {
    if (allPuzzles.length === 0) {
        showMessage('No puzzles available', 'wrong');
        return;
    }

    currentPuzzle = allPuzzles[Math.floor(Math.random() * allPuzzles.length)];

    state = {
        words: [...currentPuzzle.words],
        selected: new Set(),
        solvedGroups: [],
        mistakesLeft: 4,
        gameOver: false,
    };

    hideMessage();
    document.getElementById('solved-groups').innerHTML = '';
    document.getElementById('game-over').classList.add('hidden');
    renderMistakeDots();
    renderGrid();
}

// -- Rendering --------------------------------------------------------------

function renderGrid() {
    const grid = document.getElementById('grid');
    grid.innerHTML = '';

    state.words.forEach(word => {
        const tile = document.createElement('div');
        tile.className = 'tile' + (state.selected.has(word) ? ' selected' : '');
        tile.textContent = word;
        tile.onclick = () => toggleWord(word);
        if (state.gameOver) tile.classList.add('disabled');
        grid.appendChild(tile);
    });

    updateSubmitButton();
}

function updateSubmitButton() {
    document.getElementById('btn-submit').disabled =
        state.selected.size !== 4 || state.gameOver;
}

function renderMistakeDots() {
    const dots = [];
    for (let i = 0; i < state.mistakesLeft; i++) dots.push('\u25CF');
    document.getElementById('mistake-dots').textContent = dots.join(' ');
}

// -- Interactions -----------------------------------------------------------

function toggleWord(word) {
    if (state.gameOver) return;

    if (state.selected.has(word)) {
        state.selected.delete(word);
    } else if (state.selected.size < 4) {
        state.selected.add(word);
    }
    renderGrid();
}

function deselectAll() {
    state.selected.clear();
    renderGrid();
    hideMessage();
}

function shuffleWords() {
    for (let i = state.words.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [state.words[i], state.words[j]] = [state.words[j], state.words[i]];
    }
    renderGrid();
    hideMessage();
}

// -- Guessing ---------------------------------------------------------------

function submitGuess() {
    if (state.selected.size !== 4 || state.gameOver) return;

    const guess = new Set([...state.selected].map(w => w.toUpperCase()));

    for (const group of currentPuzzle.groups) {
        const groupWords = new Set(group.words.map(w => w.toUpperCase()));
        if (setsEqual(guess, groupWords)) {
            handleCorrectGuess(group);
            return;
        }
    }

    let bestOverlap = 0;
    for (const group of currentPuzzle.groups) {
        const groupWords = new Set(group.words.map(w => w.toUpperCase()));
        let overlap = 0;
        for (const w of guess) {
            if (groupWords.has(w)) overlap++;
        }
        bestOverlap = Math.max(bestOverlap, overlap);
    }

    if (bestOverlap === 3) {
        showMessage('One away!', 'one-away');
    } else {
        showMessage('Incorrect', 'wrong');
    }
    handleWrongGuess();
}

function setsEqual(a, b) {
    if (a.size !== b.size) return false;
    for (const item of a) {
        if (!b.has(item)) return false;
    }
    return true;
}

function handleCorrectGuess(group) {
    showMessage(group.category + '!', 'correct');

    const solvedWords = new Set(group.words.map(w => w.toUpperCase()));
    state.words = state.words.filter(w => !solvedWords.has(w.toUpperCase()));
    state.selected.clear();

    state.solvedGroups.push(group);
    renderSolvedGroup(group);
    renderGrid();

    if (state.solvedGroups.length === 4) {
        setTimeout(() => endGame(true), 600);
    }
}

function handleWrongGuess() {
    state.mistakesLeft--;
    renderMistakeDots();

    if (state.mistakesLeft <= 0) {
        endGame(false);
    }
}

function renderSolvedGroup(data) {
    const container = document.getElementById('solved-groups');
    const div = document.createElement('div');
    div.className = 'solved-group ' + (data.color || 'yellow');
    div.innerHTML =
        '<div class="category">' + escapeHtml(data.category) + '</div>' +
        '<div class="words">' + data.words.map(escapeHtml).join(', ') + '</div>';
    container.appendChild(div);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// -- Game Over --------------------------------------------------------------

function endGame(won) {
    state.gameOver = true;
    renderGrid();

    if (won) {
        document.getElementById('game-over-text').textContent = 'Congratulations!';
    } else {
        document.getElementById('game-over-text').textContent = 'Better luck next time!';
        revealRemaining();
    }
    document.getElementById('game-over').classList.remove('hidden');
}

function revealRemaining() {
    if (!currentPuzzle) return;

    const solvedCats = new Set(state.solvedGroups.map(g => g.category));
    for (const group of currentPuzzle.groups) {
        if (!solvedCats.has(group.category)) {
            renderSolvedGroup(group);
        }
    }

    state.words = [];
    renderGrid();
}

// -- Messages ---------------------------------------------------------------

function showMessage(text, type) {
    const bar = document.getElementById('message-bar');
    bar.textContent = text;
    bar.className = 'message-bar ' + type;
    clearTimeout(showMessage._timer);
    showMessage._timer = setTimeout(() => hideMessage(), 2500);
}

function hideMessage() {
    const bar = document.getElementById('message-bar');
    if (bar) bar.className = 'message-bar hidden';
}
