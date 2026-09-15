package main

import (
	"encoding/json"
	"math/rand/v2"
	"os"
)

// Daily records which quote was picked for which local date (YYYY-MM-DD).
type Daily struct {
	Date    string `json:"date"`
	QuoteID string `json:"quoteId"`
}

type WindowPos struct {
	X     int  `json:"x"`
	Y     int  `json:"y"`
	Saved bool `json:"saved"`
}

// State is the small amount of app state kept in state.json, separate from
// the quotes themselves.
type State struct {
	Daily  Daily     `json:"daily"`
	Window WindowPos `json:"window"`
}

// PickDaily returns the quote of the day and the Daily record to persist.
// It keeps prev's quote when prev is for today and that quote still exists;
// otherwise it picks a random quote, avoiding prev's quote when there is an
// alternative. ok is false when there are no quotes.
func PickDaily(quotes []Quote, prev Daily, today string, rng *rand.Rand) (Quote, Daily, bool) {
	if len(quotes) == 0 {
		return Quote{}, prev, false
	}
	if prev.Date == today {
		for _, q := range quotes {
			if q.ID == prev.QuoteID {
				return q, prev, true
			}
		}
	}
	q := pickRandom(quotes, prev.QuoteID, rng)
	return q, Daily{Date: today, QuoteID: q.ID}, true
}

// pickRandom returns a random quote whose ID is not excludeID, unless that is
// the only quote available. quotes must not be empty.
func pickRandom(quotes []Quote, excludeID string, rng *rand.Rand) Quote {
	candidates := make([]Quote, 0, len(quotes))
	for _, q := range quotes {
		if q.ID != excludeID {
			candidates = append(candidates, q)
		}
	}
	if len(candidates) == 0 {
		candidates = quotes
	}
	return candidates[rng.IntN(len(candidates))]
}

// loadState reads state.json. A missing or unreadable file yields an empty
// State: it only caches today's pick and the window position, so starting
// fresh is harmless.
func loadState(path string) State {
	data, err := os.ReadFile(path)
	if err != nil {
		return State{}
	}
	var st State
	if err := json.Unmarshal(data, &st); err != nil {
		return State{}
	}
	return st
}

func saveState(path string, st State) error {
	return writeJSONAtomic(path, st)
}
