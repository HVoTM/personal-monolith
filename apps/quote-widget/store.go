package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

const maxQuoteLen = 500

var (
	ErrEmptyQuote   = errors.New("quote text is empty")
	ErrQuoteTooLong = fmt.Errorf("quote is longer than %d characters", maxQuoteLen)
)

type Quote struct {
	ID        string    `json:"id"`
	Text      string    `json:"text"`
	Author    string    `json:"author,omitempty"`
	CreatedAt time.Time `json:"createdAt"`
}

// Store persists quotes. JSONStore is the only implementation for now; a
// SQLite-backed store can satisfy the same interface later.
type Store interface {
	List() ([]Quote, error)
	Add(text, author string) (Quote, error)
}

// seedQuotes fill a brand-new quotes file so the widget isn't empty on first run.
var seedQuotes = []Quote{
	{Text: "The unexamined life is not worth living.", Author: "Socrates"},
	{Text: "Well begun is half done.", Author: "Aristotle"},
	{Text: "A journey of a thousand miles begins with a single step.", Author: "Lao Tzu"},
}

type quotesFile struct {
	Quotes []Quote `json:"quotes"`
}

type JSONStore struct {
	mu   sync.Mutex
	path string
}

// NewJSONStore opens the quotes file at path, creating it with seedQuotes if
// it does not exist yet.
func NewJSONStore(path string) (*JSONStore, error) {
	s := &JSONStore{path: path}
	_, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		now := time.Now().UTC()
		quotes := make([]Quote, len(seedQuotes))
		for i, q := range seedQuotes {
			q.ID = newID()
			q.CreatedAt = now
			quotes[i] = q
		}
		if err := s.write(quotes); err != nil {
			return nil, err
		}
	} else if err != nil {
		return nil, err
	}
	return s, nil
}

func (s *JSONStore) List() ([]Quote, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.read()
}

func (s *JSONStore) Add(text, author string) (Quote, error) {
	text = strings.TrimSpace(text)
	author = strings.TrimSpace(author)
	if text == "" {
		return Quote{}, ErrEmptyQuote
	}
	if utf8.RuneCountInString(text) > maxQuoteLen {
		return Quote{}, ErrQuoteTooLong
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	quotes, err := s.read()
	if err != nil {
		return Quote{}, err
	}
	q := Quote{ID: newID(), Text: text, Author: author, CreatedAt: time.Now().UTC()}
	if err := s.write(append(quotes, q)); err != nil {
		return Quote{}, err
	}
	return q, nil
}

func (s *JSONStore) read() ([]Quote, error) {
	data, err := os.ReadFile(s.path)
	if err != nil {
		return nil, err
	}
	var f quotesFile
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, fmt.Errorf("parse %s: %w", s.path, err)
	}
	return f.Quotes, nil
}

func (s *JSONStore) write(quotes []Quote) error {
	return writeJSONAtomic(s.path, quotesFile{Quotes: quotes})
}

// writeJSONAtomic writes v to a temp file next to path and renames it into
// place, so a crash mid-write never leaves a truncated file behind.
func writeJSONAtomic(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func newID() string {
	b := make([]byte, 8)
	rand.Read(b) // never returns an error since Go 1.24
	return hex.EncodeToString(b)
}
