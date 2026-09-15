package main

import (
	"context"
	"log"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/wailsapp/wails/v2/pkg/runtime"
)

const (
	edgeMargin = 16
	// Wails v2 doesn't expose the screen work area, so leave room for a
	// bottom taskbar when placing the widget for the first time.
	bottomMargin = 64
)

// App holds the widget's backend. Its exported methods are bound to the
// frontend as window.go.main.App.*.
type App struct {
	ctx       context.Context
	store     Store
	statePath string

	mu    sync.Mutex // guards state and rng
	state State
	rng   *rand.Rand
}

func NewApp() (*App, error) {
	configDir, err := os.UserConfigDir()
	if err != nil {
		return nil, err
	}
	dataDir := filepath.Join(configDir, "quote-widget")
	store, err := NewJSONStore(filepath.Join(dataDir, "quotes.json"))
	if err != nil {
		return nil, err
	}
	statePath := filepath.Join(dataDir, "state.json")
	return &App{
		store:     store,
		statePath: statePath,
		state:     loadState(statePath),
		rng:       rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64())),
	}, nil
}

func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
}

// domReady moves the still-hidden window into place, then shows it.
func (a *App) domReady(ctx context.Context) {
	a.mu.Lock()
	saved := a.state.Window
	a.mu.Unlock()

	if screenW, screenH, ok := currentScreenSize(ctx); ok {
		x := screenW - windowWidth - edgeMargin
		y := screenH - windowHeight - bottomMargin
		if saved.Saved {
			// Clamp so a position saved on a bigger or since-removed monitor stays visible.
			x = max(0, min(saved.X, screenW-windowWidth))
			y = max(0, min(saved.Y, screenH-windowHeight))
		}
		runtime.WindowSetPosition(ctx, x, y)
	}
	runtime.WindowShow(ctx)
}

// currentScreenSize returns the logical size of the screen the window is on,
// falling back to the primary screen.
func currentScreenSize(ctx context.Context) (width, height int, ok bool) {
	screens, err := runtime.ScreenGetAll(ctx)
	if err != nil {
		return 0, 0, false
	}
	var primary *runtime.Screen
	for i := range screens {
		if screens[i].IsCurrent {
			return screens[i].Size.Width, screens[i].Size.Height, true
		}
		if screens[i].IsPrimary {
			primary = &screens[i]
		}
	}
	if primary == nil {
		return 0, 0, false
	}
	return primary.Size.Width, primary.Size.Height, true
}

// beforeClose remembers the window position. Wails runs it before exiting,
// including when Quit calls runtime.Quit.
func (a *App) beforeClose(ctx context.Context) (prevent bool) {
	x, y := runtime.WindowGetPosition(ctx)
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Window = WindowPos{X: x, Y: y, Saved: true}
	if err := saveState(a.statePath, a.state); err != nil {
		log.Printf("save window position: %v", err)
	}
	return false
}

// Today returns the quote of the day, or nil if there are no quotes yet.
func (a *App) Today() (*Quote, error) {
	quotes, err := a.store.List()
	if err != nil {
		return nil, err
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	q, next, ok := PickDaily(quotes, a.state.Daily, time.Now().Format(time.DateOnly), a.rng)
	if !ok {
		return nil, nil
	}
	if next != a.state.Daily {
		a.state.Daily = next
		if err := saveState(a.statePath, a.state); err != nil {
			return nil, err
		}
	}
	return &q, nil
}

// Shuffle returns a random quote other than currentID. It doesn't change the
// saved quote of the day.
func (a *App) Shuffle(currentID string) (*Quote, error) {
	quotes, err := a.store.List()
	if err != nil || len(quotes) == 0 {
		return nil, err
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	q := pickRandom(quotes, currentID, a.rng)
	return &q, nil
}

func (a *App) AddQuote(text, author string) (Quote, error) {
	return a.store.Add(text, author)
}

func (a *App) Quit() {
	runtime.Quit(a.ctx)
}
