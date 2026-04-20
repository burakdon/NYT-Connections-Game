const state = {
  puzzle: null,
  tiles: [],
  selected: new Set(),
  solved: new Set(),
  mistakes: 0,
  locked: false,
  latestTrace: null
};

const groupColorClass = ["group-yellow", "group-green", "group-blue", "group-purple"];

const board = document.querySelector("#board");
const solvedGroups = document.querySelector("#solvedGroups");
const gameMessage = document.querySelector("#gameMessage");
const mistakeCount = document.querySelector("#mistakeCount");
const bankStatus = document.querySelector("#bankStatus");
const agentTrace = document.querySelector("#agentTrace");
const generateResult = document.querySelector("#generateResult");
const groupGenerateResult = document.querySelector("#groupGenerateResult");
const guardStatus = document.querySelector("#guardStatus");

function shuffle(items) {
  const copy = [...items];
  for (let index = copy.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [copy[index], copy[swapIndex]] = [copy[swapIndex], copy[index]];
  }
  return copy;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "content-type": "application/json" },
    ...options
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Request failed.");
  }

  return payload;
}

function setMessage(message) {
  gameMessage.textContent = message;
}

function hydrateTiles(puzzle) {
  const tiles = [];
  puzzle.groups.forEach((group, groupIndex) => {
    group.words.forEach((word) => {
      tiles.push({
        id: `${groupIndex}:${word}`,
        word,
        groupIndex
      });
    });
  });
  return shuffle(tiles);
}

async function loadBankStatus() {
  try {
    const payload = await fetchJson("/api/puzzles");
    const groupText = payload.group_count ? `, ${payload.group_count} groups ready` : "";
    bankStatus.textContent = `${payload.count} puzzles in the bank${groupText}`;
  } catch (error) {
    bankStatus.textContent = "Puzzle bank unavailable";
  }
}

async function loadGuardStatus() {
  try {
    const payload = await fetchJson("/api/nyt-guard");
    if (payload.ready) {
      guardStatus.textContent =
        `NYT guard ready: ${payload.board_count} board hash(es), ` +
        `${payload.group_set_count} group-set hash(es).`;
    } else {
      guardStatus.textContent =
        "NYT guard needs archive hashes before generated puzzles can be accepted.";
    }
  } catch (error) {
    guardStatus.textContent = "NYT guard status unavailable.";
  }
}

async function loadRandomPuzzle() {
  state.locked = false;
  state.selected = new Set();
  state.solved = new Set();
  state.mistakes = 0;

  try {
    const payload = await fetchJson("/api/puzzles/random");
    state.puzzle = payload.puzzle;
    state.tiles = hydrateTiles(state.puzzle);
    setMessage("Choose four words, then submit.");
  } catch (error) {
    setMessage(error.message);
  }

  renderGame();
  loadBankStatus();
}

function renderGame() {
  mistakeCount.textContent = `${state.mistakes} / 4`;
  renderSolvedGroups();
  renderBoard();
}

function renderSolvedGroups() {
  solvedGroups.innerHTML = "";

  if (!state.puzzle) {
    return;
  }

  state.puzzle.groups.forEach((group, groupIndex) => {
    if (!state.solved.has(groupIndex)) {
      return;
    }

    const element = document.createElement("article");
    element.className = `solved-group ${groupColorClass[groupIndex] || "group-green"}`;
    element.innerHTML = `
      <h3>${escapeHtml(group.category)}</h3>
      <p>${group.words.map(escapeHtml).join(", ")}</p>
    `;
    solvedGroups.appendChild(element);
  });
}

function renderBoard() {
  board.innerHTML = "";

  if (!state.puzzle) {
    return;
  }

  const activeTiles = state.tiles.filter((tile) => !state.solved.has(tile.groupIndex));

  activeTiles.forEach((tile) => {
    const button = document.createElement("button");
    button.className = "tile";
    button.type = "button";
    button.textContent = tile.word;
    button.disabled = state.locked;

    if (state.selected.has(tile.id)) {
      button.classList.add("is-selected");
    }

    button.addEventListener("click", () => toggleTile(tile.id));
    board.appendChild(button);
  });
}

function getActiveTiles() {
  return state.tiles.filter((tile) => !state.solved.has(tile.groupIndex));
}

function getSelectedActiveTiles() {
  const activeTileIds = new Set(getActiveTiles().map((tile) => tile.id));
  state.selected = new Set([...state.selected].filter((tileId) => activeTileIds.has(tileId)));
  return getActiveTiles().filter((tile) => state.selected.has(tile.id));
}

function isOneAway(selectedTiles) {
  const counts = new Map();
  selectedTiles.forEach((tile) => {
    counts.set(tile.groupIndex, (counts.get(tile.groupIndex) || 0) + 1);
  });
  return [...counts.values()].some((count) => count === 3);
}

function toggleTile(tileId) {
  if (state.locked) {
    return;
  }

  if (state.selected.has(tileId)) {
    state.selected.delete(tileId);
  } else if (state.selected.size < 4) {
    state.selected.add(tileId);
  } else {
    setMessage("You can only select four words.");
  }

  renderBoard();
}

function submitGuess() {
  if (!state.puzzle || state.locked) {
    return;
  }

  if (state.selected.size !== 4) {
    setMessage("Select exactly four words.");
    return;
  }

  const selectedTiles = getSelectedActiveTiles();

  if (selectedTiles.length !== 4) {
    state.selected = new Set();
    setMessage("That selection changed. Try those four again.");
    renderGame();
    return;
  }

  const groupIndex = selectedTiles[0].groupIndex;
  const correct = selectedTiles.every((tile) => tile.groupIndex === groupIndex);
  const oneAway = !correct && isOneAway(selectedTiles);

  if (correct && !state.solved.has(groupIndex)) {
    state.solved.add(groupIndex);
    state.selected = new Set();
    const category = state.puzzle.groups[groupIndex].category;
    setMessage(`Correct: ${category}.`);

    if (state.solved.size === 4) {
      state.locked = true;
      setMessage("All four groups found.");
    }
  } else {
    state.mistakes += 1;
    state.selected = new Set();

    if (state.mistakes >= 4) {
      revealAll();
      setMessage("Four mistakes. Answers revealed.");
    } else if (oneAway) {
      setMessage(`One away. ${4 - state.mistakes} mistake${4 - state.mistakes === 1 ? "" : "s"} left.`);
    } else {
      setMessage(`${4 - state.mistakes} mistake${4 - state.mistakes === 1 ? "" : "s"} left.`);
    }
  }

  renderGame();
}

function revealAll() {
  if (!state.puzzle) {
    return;
  }

  state.locked = true;
  state.puzzle.groups.forEach((_, index) => state.solved.add(index));
  state.selected = new Set();
}

function clearSelection() {
  state.selected = new Set();
  setMessage("Selection cleared.");
  renderBoard();
}

function shuffleActiveWords() {
  const solvedTiles = state.tiles.filter((tile) => state.solved.has(tile.groupIndex));
  const activeTiles = state.tiles.filter((tile) => !state.solved.has(tile.groupIndex));
  state.tiles = [...solvedTiles, ...shuffle(activeTiles)];
  renderBoard();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.tab === name);
  });

  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("is-active", panel.id === name);
  });

  if (name === "lab") {
    loadLatestTrace();
  }
}

async function runAgents(event) {
  event.preventDefault();

  const count = Number(document.querySelector("#generateCount").value || 1);
  const difficulty = document.querySelector("#generateDifficulty").value;
  const strategy = "group-bank";
  const save = document.querySelector("#saveGenerated").checked;

  generateResult.textContent = "Building puzzles from approved groups...";

  try {
    const payload = await fetchJson("/api/generate", {
      method: "POST",
      body: JSON.stringify({ count, difficulty, strategy, save })
    });

    state.latestTrace = payload;
    renderTrace(payload.trace || []);
    generateResult.textContent =
      `Accepted ${payload.accepted.length} puzzle(s), rejected ${payload.rejected.length}. ` +
      `${payload.saved ? "Saved to the bank." : "Not saved."}`;

    await loadBankStatus();
    await loadGuardStatus();
    switchTab("lab");
  } catch (error) {
    generateResult.textContent = error.message;
  }
}

async function runGroupAgents(event) {
  event.preventDefault();

  const count = Number(document.querySelector("#groupGenerateCount").value || 1);
  const difficulty = document.querySelector("#groupGenerateDifficulty").value;
  const theme = document.querySelector("#groupGenerateTheme").value.trim();
  const save = document.querySelector("#saveGeneratedGroups").checked;

  groupGenerateResult.textContent = "Group agents are working...";

  try {
    const payload = await fetchJson("/api/generate-groups", {
      method: "POST",
      body: JSON.stringify({ count, difficulty, theme, save })
    });

    state.latestTrace = payload;
    renderTrace(payload.trace || []);
    groupGenerateResult.textContent =
      `Accepted ${payload.accepted.length} group(s), rejected ${payload.rejected.length}. ` +
      `${payload.saved ? "Saved to the group bank." : "Not saved."}`;

    await loadBankStatus();
    switchTab("lab");
  } catch (error) {
    groupGenerateResult.textContent = error.message;
  }
}

async function loadLatestTrace() {
  if (state.latestTrace) {
    renderTrace(state.latestTrace.trace || []);
    return;
  }

  try {
    const payload = await fetchJson("/api/agent-runs/latest");
    renderTrace(payload.trace || []);
  } catch (error) {
    agentTrace.innerHTML = `<div class="result-box">${escapeHtml(error.message)}</div>`;
  }
}

function renderTrace(trace) {
  agentTrace.innerHTML = "";

  if (!trace.length) {
    agentTrace.innerHTML = '<div class="result-box">No agent run yet.</div>';
    return;
  }

  trace.forEach((event) => {
    const element = document.createElement("article");
    element.className = "agent-event";
    const details = event.details?.raw_excerpt || JSON.stringify(event.details || {}, null, 2);
    element.innerHTML = `
      <h3>${escapeHtml(event.agent)}</h3>
      <small>${escapeHtml(event.status)} - ${escapeHtml(event.duration_seconds)}s</small>
      <p>${escapeHtml(event.summary)}</p>
      <pre>${escapeHtml(details)}</pre>
    `;
    agentTrace.appendChild(element);
  });
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => switchTab(tab.dataset.tab));
});

document.querySelector("#submitGuess").addEventListener("click", submitGuess);
document.querySelector("#shuffleWords").addEventListener("click", shuffleActiveWords);
document.querySelector("#clearSelection").addEventListener("click", clearSelection);
document.querySelector("#newPuzzle").addEventListener("click", loadRandomPuzzle);
document.querySelector("#generateForm").addEventListener("submit", runAgents);
document.querySelector("#groupGenerateForm").addEventListener("submit", runGroupAgents);

loadRandomPuzzle();
loadLatestTrace();
loadGuardStatus();
