package main

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

func TestNewJSONStoreSeedsOnFirstRun(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "quotes.json")
	s, err := NewJSONStore(path)
	if err != nil {
		t.Fatal(err)
	}
	quotes, err := s.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(quotes) != len(seedQuotes) {
		t.Fatalf("got %d quotes, want %d seeded", len(quotes), len(seedQuotes))
	}
	for _, q := range quotes {
		if q.ID == "" || q.CreatedAt.IsZero() {
			t.Errorf("seeded quote missing ID or CreatedAt: %+v", q)
		}
	}
}

func TestAddRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "quotes.json")
	s, err := NewJSONStore(path)
	if err != nil {
		t.Fatal(err)
	}
	added, err := s.Add("  Stay hungry, stay foolish.  ", "  Me ")
	if err != nil {
		t.Fatal(err)
	}
	if added.Text != "Stay hungry, stay foolish." || added.Author != "Me" {
		t.Errorf("Add did not trim input: %+v", added)
	}

	// Reopening must not re-seed and must still contain the added quote.
	reopened, err := NewJSONStore(path)
	if err != nil {
		t.Fatal(err)
	}
	quotes, err := reopened.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(quotes) != len(seedQuotes)+1 {
		t.Fatalf("got %d quotes after reopen, want %d", len(quotes), len(seedQuotes)+1)
	}
	if last := quotes[len(quotes)-1]; last.ID != added.ID || last.Text != added.Text {
		t.Errorf("last quote = %+v, want %+v", last, added)
	}
}

func TestAddRejectsInvalidText(t *testing.T) {
	s, err := NewJSONStore(filepath.Join(t.TempDir(), "quotes.json"))
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		text string
		want error
	}{
		{"empty", "", ErrEmptyQuote},
		{"whitespace", " \n\t ", ErrEmptyQuote},
		{"too long", strings.Repeat("a", maxQuoteLen+1), ErrQuoteTooLong},
		{"at limit", strings.Repeat("é", maxQuoteLen), nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := s.Add(tt.text, ""); !errors.Is(err, tt.want) {
				t.Errorf("Add error = %v, want %v", err, tt.want)
			}
		})
	}
}
