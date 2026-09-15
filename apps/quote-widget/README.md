# quote-widget

A small always-on-top desktop widget that shows a quote of the day, picked from quotes you add yourself.

Built with Go + [Wails v2](https://wails.io). The UI is plain HTML/CSS/JS in `frontend/dist`, with no Node build step.

## Usage

- **Drag** the card to move it. Its position is remembered.
- **Hover** to show the toolbar:
  - **+** adds a quote. Press Ctrl+Enter to save, Esc to cancel.
  - **⟳** shows another random quote. Today's pick doesn't change.
  - **✕** quits.
- The quote of the day stays the same all day and changes after midnight. It avoids repeating yesterday's quote.

## Data

Stored as JSON in `%APPDATA%\quote-widget\`:

- `quotes.json`: your quotes. The first run seeds it with 3 starter quotes.
- `state.json`: today's pick and the window position. You can delete it safely.

Quotes are accessed through the `Store` interface in `store.go`. A SQLite store can replace `JSONStore` without touching the rest of the app.

## Development

Prerequisites: Go 1.27+, the Wails CLI (`go install github.com/wailsapp/wails/v2/cmd/wails@latest`), and WebView2 (built into Windows 11).

```sh
cd apps/quote-widget
go test ./...     # unit tests
wails dev         # run with live reload
wails build       # produces build/bin/quote-widget.exe
```

To start the widget at login, put a shortcut to `quote-widget.exe` in `shell:startup`.
