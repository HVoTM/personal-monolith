// Go methods bound by Wails (see app.go) are exposed on window.go.main.App.
const api = () => window.go.main.App;
const $ = (id) => document.getElementById(id);

let current = null;
let loadedDay = "";

function showQuote(quote) {
  current = quote;
  const text = $("quote-text");
  if (!quote) {
    text.textContent = "No quotes yet — hover and press +";
    text.title = "";
    $("quote-author").textContent = "";
    return;
  }
  text.textContent = quote.text;
  text.title = quote.text;
  $("quote-author").textContent = quote.author ? `— ${quote.author}` : "";
}

function showForm(open) {
  $("quote-view").hidden = open;
  $("add-view").hidden = !open;
  $("error").textContent = "";
  if (open) {
    $("text-input").value = "";
    $("author-input").value = "";
    $("text-input").focus();
  }
}

async function loadToday() {
  loadedDay = new Date().toDateString();
  try {
    showQuote(await api().Today());
  } catch (err) {
    $("quote-text").textContent = `Couldn't load quotes: ${err}`;
  }
}

$("shuffle-btn").addEventListener("click", async () => {
  try {
    showQuote(await api().Shuffle(current ? current.id : ""));
  } catch (err) {
    $("quote-text").textContent = `Couldn't shuffle: ${err}`;
  }
});

$("add-btn").addEventListener("click", () => showForm(true));
$("cancel-btn").addEventListener("click", () => showForm(false));
$("quit-btn").addEventListener("click", () => api().Quit());

$("add-view").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const quote = await api().AddQuote($("text-input").value, $("author-input").value);
    showForm(false);
    showQuote(quote);
  } catch (err) {
    $("error").textContent = String(err);
  }
});

$("add-view").addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    showForm(false);
  } else if (event.key === "Enter" && event.ctrlKey) {
    $("add-view").requestSubmit();
  }
});

// The widget can stay open for days: pick up the new quote after midnight,
// unless the add form is open.
setInterval(() => {
  if (new Date().toDateString() !== loadedDay && $("add-view").hidden) {
    loadToday();
  }
}, 60_000);

loadToday();
