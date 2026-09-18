/**
 * The widget's side of the postMessage channel to the parent page
 * (i.e. the SDK running in the customer's website).
 *
 * Security invariants — mirrors sdk/src/iframe.ts's rules from the
 * other side of the channel:
 * - Outgoing messages are sent to a specific target origin, never
 *   `"*"`. The target is derived from `document.referrer`'s origin
 *   (the actual embedding page), not from anything the message
 *   content itself claims.
 * - Incoming messages are only accepted if `event.origin` matches
 *   that same parent origin, checked before the payload is trusted.
 * - If the widget is ever loaded stand-alone (not inside an iframe,
 *   `window.parent === window`), none of this messaging is attempted
 *   at all — there is no parent to talk to.
 */

const WIDGET_MESSAGE_SOURCE = "room-visualizer-widget" as const;
const SDK_MESSAGE_SOURCE = "room-visualizer-sdk" as const;

export type WidgetToSdkMessage =
  | { source: typeof WIDGET_MESSAGE_SOURCE; type: "ready" }
  | { source: typeof WIDGET_MESSAGE_SOURCE; type: "resize"; height: number }
  | {
      source: typeof WIDGET_MESSAGE_SOURCE;
      type: "render-requested";
      payload: { materialId: string | null };
    }
  | { source: typeof WIDGET_MESSAGE_SOURCE; type: "error"; message: string };

export interface SdkConfigPayload {
  tenantId: string;
  apiBaseUrl: string;
}

function isEmbedded(): boolean {
  return window.parent !== window;
}

function getParentOrigin(): string | null {
  if (!document.referrer) {
    return null;
  }
  try {
    return new URL(document.referrer).origin;
  } catch {
    return null;
  }
}

/**
 * Send a message to the parent SDK, if and only if the widget is
 * actually embedded and a parent origin could be determined. Silently
 * no-ops otherwise (e.g. when the widget is opened stand-alone during
 * local development) rather than throwing — messaging failures should
 * never break the widget's own functionality.
 */
export function sendToParent(message: WidgetToSdkMessage): void {
  if (!isEmbedded()) {
    return;
  }

  const parentOrigin = getParentOrigin();
  if (parentOrigin === null) {
    return;
  }

  window.parent.postMessage(message, parentOrigin);
}

export function notifyReady(): void {
  sendToParent({ source: WIDGET_MESSAGE_SOURCE, type: "ready" });
}

export function notifyResize(height: number): void {
  sendToParent({ source: WIDGET_MESSAGE_SOURCE, type: "resize", height });
}

export function notifyRenderRequested(materialId: string | null): void {
  sendToParent({
    source: WIDGET_MESSAGE_SOURCE,
    type: "render-requested",
    payload: { materialId },
  });
}

export function notifyError(message: string): void {
  sendToParent({ source: WIDGET_MESSAGE_SOURCE, type: "error", message });
}

/**
 * Listen for the SDK's config message. Validates `event.origin`
 * against the actual parent origin before accepting anything —
 * exactly mirroring the SDK's own validation of messages coming from
 * the widget.
 */
export function listenForSdkConfig(onConfig: (payload: SdkConfigPayload) => void): () => void {
  const parentOrigin = getParentOrigin();

  const handleMessage = (event: MessageEvent): void => {
    if (!isEmbedded()) {
      return;
    }

    if (parentOrigin === null || event.origin !== parentOrigin) {
      return;
    }

    if (event.source !== window.parent) {
      return;
    }

    const data = event.data as Record<string, unknown> | null;
    if (
      typeof data !== "object" ||
      data === null ||
      data.source !== SDK_MESSAGE_SOURCE ||
      data.type !== "config"
    ) {
      return;
    }

    const payload = data.payload as Record<string, unknown> | undefined;
    if (
      typeof payload !== "object" ||
      payload === null ||
      typeof payload.tenantId !== "string" ||
      typeof payload.apiBaseUrl !== "string"
    ) {
      return;
    }

    onConfig({ tenantId: payload.tenantId, apiBaseUrl: payload.apiBaseUrl });
  };

  window.addEventListener("message", handleMessage);
  return () => window.removeEventListener("message", handleMessage);
}
