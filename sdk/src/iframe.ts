/**
 * Iframe creation, injection, and the origin-validated postMessage
 * channel to it.
 *
 * Security invariants:
 * - The iframe `src` is built from `widgetOrigin` (which the
 *   INTEGRATOR configured, not attacker-influenceable) plus only
 *   `tenantId` as a query parameter. No credential ever goes into
 *   this URL.
 * - `sandbox` is set to the minimum needed: `allow-scripts` (the
 *   widget is a React app), `allow-forms` (the upload UI uses a file
 *   input), `allow-same-origin` (needed for the widget's own `fetch`
 *   calls to the API to succeed under the sandbox). Deliberately
 *   NOT included: `allow-top-navigation`, `allow-popups`,
 *   `allow-modals`, `allow-pointer-lock` — the widget has no
 *   legitimate need for any of them, and omitting them means even a
 *   fully compromised widget bundle couldn't hijack the parent page
 *   or spawn arbitrary browser UI.
 * - Incoming messages are checked against `event.origin === widgetOrigin`
 *   BEFORE the message's `data` is even passed to `isWidgetMessage`.
 *   `event.origin` is a property the browser itself sets from the
 *   actual sending frame's origin — a malicious script cannot forge
 *   it, unlike anything inside `event.data`.
 * - Outgoing messages always specify `widgetOrigin` as the explicit
 *   target (never `"*"`), so a message can't be read by some other
 *   origin if the iframe ever unexpectedly navigates elsewhere.
 */

import { isWidgetMessage, SDK_MESSAGE_SOURCE, type WidgetToSdkMessage } from "./messages";
import type { ResolvedRoomVisualizerConfig } from "./config";

export interface WidgetHandle {
  /** Remove the iframe and tear down the message listener. */
  destroy(): void;
}

export interface WidgetCallbacks {
  onReady?: () => void;
  onResize?: (height: number) => void;
  onRenderRequested?: (materialId: string | null) => void;
  onError?: (message: string) => void;
}

export function createWidget(
  config: ResolvedRoomVisualizerConfig,
  callbacks: WidgetCallbacks = {},
): WidgetHandle {
  const iframe = document.createElement("iframe");

  const embedUrl = new URL("/embed", config.widgetOrigin);
  embedUrl.searchParams.set("tenantId", config.tenantId);
  iframe.src = embedUrl.toString();

  // Minimum sandbox permissions — see module docstring.
  iframe.setAttribute("sandbox", "allow-scripts allow-forms allow-same-origin");
  iframe.style.border = "none";
  iframe.style.width = "100%";
  iframe.style.minHeight = "480px";
  iframe.setAttribute("title", "Room Visualizer");

  config.container.appendChild(iframe);

  const handleMessage = (event: MessageEvent): void => {
    // Origin check FIRST, before touching event.data at all. This is
    // the load-bearing check: event.origin is browser-guaranteed and
    // cannot be spoofed by the sending page's script, unlike anything
    // inside the message payload itself.
    if (event.origin !== config.widgetOrigin) {
      return;
    }

    if (event.source !== iframe.contentWindow) {
      // Defensive: ignore messages from any frame other than the one
      // we created, even if the origin happened to match.
      return;
    }

    if (!isWidgetMessage(event.data)) {
      return;
    }

    dispatchWidgetMessage(event.data, callbacks);
  };

  window.addEventListener("message", handleMessage);

  iframe.addEventListener("load", () => {
    // Send the widget its config as soon as the iframe document has
    // loaded. tenantId already travelled via the URL (it's public),
    // but sending it again here over the validated postMessage
    // channel lets the widget code use a single code path regardless
    // of how it's embedded, without re-parsing its own URL.
    iframe.contentWindow?.postMessage(
      {
        source: SDK_MESSAGE_SOURCE,
        type: "config",
        payload: { tenantId: config.tenantId, apiBaseUrl: config.apiBaseUrl },
      },
      config.widgetOrigin,
    );
  });

  return {
    destroy(): void {
      window.removeEventListener("message", handleMessage);
      iframe.remove();
    },
  };
}

function dispatchWidgetMessage(message: WidgetToSdkMessage, callbacks: WidgetCallbacks): void {
  switch (message.type) {
    case "ready":
      callbacks.onReady?.();
      break;
    case "resize":
      callbacks.onResize?.(message.height);
      break;
    case "render-requested":
      callbacks.onRenderRequested?.(message.payload.materialId);
      break;
    case "error":
      callbacks.onError?.(message.message);
      break;
  }
}
