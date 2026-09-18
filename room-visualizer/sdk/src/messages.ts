/**
 * Typed postMessage protocol between the SDK (parent page) and the
 * widget (iframe).
 *
 * Security invariants enforced here:
 * - No message in either direction ever carries a secret/API key —
 *   only public, display-safe data (tenantId, a render-request's
 *   material selection, resize dimensions, error text).
 * - Every message has a `source` discriminator that must be checked
 *   BEFORE trusting anything else in the payload — see
 *   `isWidgetMessage` / `isSdkMessage`. This stops a message from an
 *   unrelated script (same window, different purpose) from being
 *   misinterpreted as SDK/widget traffic.
 * - Origin validation is NOT done here — it can't be, since a
 *   MessageEvent's origin isn't part of the message data itself. It's
 *   done by the caller (iframe.ts / the widget's parentMessaging.ts)
 *   using `event.origin`, which is a browser-guaranteed field that
 *   scripts cannot spoof.
 */

export const WIDGET_MESSAGE_SOURCE = "room-visualizer-widget" as const;
export const SDK_MESSAGE_SOURCE = "room-visualizer-sdk" as const;

export type WidgetToSdkMessage =
  | { source: typeof WIDGET_MESSAGE_SOURCE; type: "ready" }
  | { source: typeof WIDGET_MESSAGE_SOURCE; type: "resize"; height: number }
  | {
      source: typeof WIDGET_MESSAGE_SOURCE;
      type: "render-requested";
      payload: { materialId: string | null };
    }
  | { source: typeof WIDGET_MESSAGE_SOURCE; type: "error"; message: string };

export type SdkToWidgetMessage = {
  source: typeof SDK_MESSAGE_SOURCE;
  type: "config";
  payload: { tenantId: string; apiBaseUrl: string };
};

/**
 * Type guard: is `data` a well-formed message from the widget?
 * Checks the discriminator first, then the shape for each known type.
 * Anything that doesn't match is treated as untrusted noise and
 * ignored by the caller — never partially trusted.
 */
export function isWidgetMessage(data: unknown): data is WidgetToSdkMessage {
  if (typeof data !== "object" || data === null) {
    return false;
  }

  const candidate = data as Record<string, unknown>;
  if (candidate.source !== WIDGET_MESSAGE_SOURCE) {
    return false;
  }

  switch (candidate.type) {
    case "ready":
      return true;
    case "resize":
      return typeof candidate.height === "number";
    case "render-requested": {
      const payload = candidate.payload as Record<string, unknown> | undefined;
      return (
        typeof payload === "object" &&
        payload !== null &&
        (payload.materialId === null || typeof payload.materialId === "string")
      );
    }
    case "error":
      return typeof candidate.message === "string";
    default:
      return false;
  }
}

/** Same idea, for messages arriving in the widget from the SDK. */
export function isSdkMessage(data: unknown): data is SdkToWidgetMessage {
  if (typeof data !== "object" || data === null) {
    return false;
  }

  const candidate = data as Record<string, unknown>;
  if (candidate.source !== SDK_MESSAGE_SOURCE || candidate.type !== "config") {
    return false;
  }

  const payload = candidate.payload as Record<string, unknown> | undefined;
  return (
    typeof payload === "object" &&
    payload !== null &&
    typeof payload.tenantId === "string" &&
    typeof payload.apiBaseUrl === "string"
  );
}
