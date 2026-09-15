package main

import (
	"embed"
	"log"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
	"github.com/wailsapp/wails/v2/pkg/options/windows"
)

//go:embed all:frontend/dist
var assets embed.FS

const (
	windowWidth  = 340
	windowHeight = 180
)

func main() {
	app, err := NewApp()
	if err != nil {
		log.Fatal(err)
	}

	err = wails.Run(&options.App{
		Title:         "Quote of the Day",
		Width:         windowWidth,
		Height:        windowHeight,
		DisableResize: true,
		Frameless:     true,
		AlwaysOnTop:   true,
		// Shown in domReady once positioned, so it never flashes at screen centre.
		StartHidden:      true,
		BackgroundColour: &options.RGBA{R: 0, G: 0, B: 0, A: 0},
		AssetServer:      &assetserver.Options{Assets: assets},
		OnStartup:        app.startup,
		OnDomReady:       app.domReady,
		OnBeforeClose:    app.beforeClose,
		Bind:             []interface{}{app},
		SingleInstanceLock: &options.SingleInstanceLock{
			UniqueId: "quote-widget-hvotm",
		},
		Windows: &windows.Options{
			WebviewIsTransparent:              true,
			DisableFramelessWindowDecorations: true,
		},
	})
	if err != nil {
		log.Fatal(err)
	}
}
