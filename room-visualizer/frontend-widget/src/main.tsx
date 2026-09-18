import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { EmbedApp } from "./EmbedApp";

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("Root element not found");
}

// No router dependency added for a single-route split (per the
// approved default) — the widget only ever needs to distinguish
// "/embed" (the real screen, run inside the SDK's iframe) from
// anything else (a bare landing stub, useful when someone opens the
// widget's own URL directly rather than through the SDK).
const isEmbedRoute = window.location.pathname.startsWith("/embed");

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>{isEmbedRoute ? <EmbedApp /> : <App />}</React.StrictMode>,
);
