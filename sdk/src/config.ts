/**
 * SDK configuration types and validation.
 *
 * Security note: `RoomVisualizerConfig` intentionally has no field for
 * an API key, secret, or credential of any kind. `tenantId` is the
 * ONLY tenant-identifying value the SDK ever handles, and it is public
 * information (it's returned in the tenant-creation response and used
 * throughout the internal API's URLs) — not a secret. If a future
 * change to this file ever adds a field that looks like a credential,
 * that is itself a sign something has gone wrong with this milestone's
 * core security property.
 */

const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export interface RoomVisualizerConfig {
  /** Public tenant identifier (a UUID). NOT a secret. */
  tenantId: string;
  /** CSS selector or a real HTMLElement to inject the iframe into. */
  container: string | HTMLElement;
  /**
   * Origin the widget is hosted at, e.g. "https://widget.roomvisualizer.com".
   * Defaults to the production widget origin. Override for local
   * development against a locally-served widget build.
   */
  widgetOrigin?: string;
  /**
   * Base URL of the backend API, e.g. "https://api.roomvisualizer.com".
   * Passed through to the widget (via the iframe URL) so it knows
   * where to call the public endpoints. Defaults to the production API.
   */
  apiBaseUrl?: string;
}

export interface ResolvedRoomVisualizerConfig {
  tenantId: string;
  container: HTMLElement;
  widgetOrigin: string;
  apiBaseUrl: string;
}

const DEFAULT_WIDGET_ORIGIN = "https://widget.roomvisualizer.com";
const DEFAULT_API_BASE_URL = "https://api.roomvisualizer.com";

export class RoomVisualizerConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RoomVisualizerConfigError";
  }
}

/**
 * Validate and normalize a caller-supplied config, resolving the
 * container selector to a real element and filling in defaults.
 *
 * Fails loudly (throws) on any invalid input, per the approved design
 * — a misconfigured SDK should never silently no-op, since a customer
 * integrating it would have no way to know their widget simply never
 * appeared.
 */
export function resolveConfig(
  config: RoomVisualizerConfig,
): ResolvedRoomVisualizerConfig {
  if (!config || typeof config !== "object") {
    throw new RoomVisualizerConfigError(
      "RoomVisualizer.init() requires a configuration object.",
    );
  }

  if (typeof config.tenantId !== "string" || !UUID_PATTERN.test(config.tenantId)) {
    throw new RoomVisualizerConfigError(
      "RoomVisualizer.init() requires a valid `tenantId` (UUID string). " +
        "This must be the PUBLIC tenant id, never a secret API key.",
    );
  }

  const container = resolveContainer(config.container);

  const widgetOrigin = normalizeOrigin(config.widgetOrigin ?? DEFAULT_WIDGET_ORIGIN);
  const apiBaseUrl = config.apiBaseUrl ?? DEFAULT_API_BASE_URL;

  return {
    tenantId: config.tenantId,
    container,
    widgetOrigin,
    apiBaseUrl,
  };
}

function resolveContainer(container: string | HTMLElement | undefined): HTMLElement {
  if (container instanceof HTMLElement) {
    return container;
  }

  if (typeof container === "string") {
    const element = document.querySelector(container);
    if (element instanceof HTMLElement) {
      return element;
    }
    throw new RoomVisualizerConfigError(
      `RoomVisualizer.init() could not find a container element matching "${container}".`,
    );
  }

  throw new RoomVisualizerConfigError(
    "RoomVisualizer.init() requires `container` to be a CSS selector string or an HTMLElement.",
  );
}

function normalizeOrigin(origin: string): string {
  // Strip any trailing slash so postMessage target-origin comparisons
  // are exact and predictable (browsers report event.origin without a
  // trailing slash).
  return origin.replace(/\/+$/, "");
}
