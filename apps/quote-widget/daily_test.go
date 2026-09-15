package main

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
)

var testQuotes = []Quote{
	{ID: "a", Text: "A"},
	{ID: "b", Text: "B"},
	{ID: "c", Text: "C"},
}

func newRNG(seed uint64) *rand.Rand {
	return rand.New(rand.NewPCG(seed, seed))
}

func TestPickDailyEmpty(t *testing.T) {
	if _, _, ok := PickDaily(nil, Daily{}, "2026-09-15", newRNG(1)); ok {
		t.Error("PickDaily with no quotes returned ok=true")
	}
}

func TestPickDailyKeepsSameDay(t *testing.T) {
	prev := Daily{Date: "2026-09-15", QuoteID: "b"}
	for seed := range uint64(20) {
		q, next, ok := PickDaily(testQuotes, prev, "2026-09-15", newRNG(seed))
		if !ok || q.ID != "b" || next != prev {
			t.Fatalf("seed %d: got %+v %+v %v, want quote b and unchanged Daily", seed, q, next, ok)
		}
	}
}

func TestPickDailyNewDayAvoidsYesterday(t *testing.T) {
	prev := Daily{Date: "2026-09-14", QuoteID: "b"}
	for seed := range uint64(50) {
		q, next, ok := PickDaily(testQuotes, prev, "2026-09-15", newRNG(seed))
		if !ok || q.ID == "b" {
			t.Fatalf("seed %d: picked %q, want anything but yesterday's b", seed, q.ID)
		}
		if next != (Daily{Date: "2026-09-15", QuoteID: q.ID}) {
			t.Fatalf("seed %d: next = %+v, want today with %q", seed, next, q.ID)
		}
	}
}

func TestPickDailyRepicksDeletedQuote(t *testing.T) {
	q, next, ok := PickDaily(testQuotes, Daily{Date: "2026-09-15", QuoteID: "gone"}, "2026-09-15", newRNG(1))
	if !ok || q.ID == "" || next.QuoteID != q.ID {
		t.Fatalf("got %+v %+v %v, want a fresh pick recorded in next", q, next, ok)
	}
}

func TestPickDailySingleQuoteRepeats(t *testing.T) {
	only := testQuotes[:1]
	q, _, ok := PickDaily(only, Daily{Date: "2026-09-14", QuoteID: "a"}, "2026-09-15", newRNG(1))
	if !ok || q.ID != "a" {
		t.Fatalf("got %+v %v, want the only quote a", q, ok)
	}
}

func TestLoadState(t *testing.T) {
	dir := t.TempDir()

	if st := loadState(filepath.Join(dir, "missing.json")); st != (State{}) {
		t.Errorf("missing file: got %+v, want empty state", st)
	}

	corrupt := filepath.Join(dir, "corrupt.json")
	if err := os.WriteFile(corrupt, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if st := loadState(corrupt); st != (State{}) {
		t.Errorf("corrupt file: got %+v, want empty state", st)
	}

	path := filepath.Join(dir, "state.json")
	want := State{
		Daily:  Daily{Date: "2026-09-15", QuoteID: "b"},
		Window: WindowPos{X: 100, Y: 200, Saved: true},
	}
	if err := saveState(path, want); err != nil {
		t.Fatal(err)
	}
	if got := loadState(path); got != want {
		t.Errorf("round trip: got %+v, want %+v", got, want)
	}
}
