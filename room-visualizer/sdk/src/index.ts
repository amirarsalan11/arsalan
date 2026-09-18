/**
 * Room Visualizer JS SDK — public entry point.
 *
 * SDK + Widget milestone: replaces the Milestone 1 stub with a real
 * implementation. Responsibilities (per architecture lock):
 * - Create the iframe.
 * - Inject the iframe into the customer's page.
 * - Pass configuration to the widget.
 * - Communicate with the parent website via window.postMessage,
 *   validating message origin.
 * - Handle widget lifecycle (destroy()).
 *
 * The SDK remains framework independent: no React, no AI logic, no
 * image-processing logic — see frontend-widget/ for all of that.
 *
 * Security: this module (and everything it imports) never accepts,
 * stores, or transmits an API key/secret. Only the public `tenantId`
 * crosses the SDK -> iframe boundary. See config.ts and iframe.ts for
 * the specific invariants that enforce this.
 */

import { resolveConfig, type RoomVisualizerConfig } from "./config";
import { createWidget, type WidgetCallbacks, type WidgetHandle } from "./iframe";

export const SDK_VERSION = "0.2.0";

export type { RoomVisualizerConfig, WidgetCallbacks, WidgetHandle };
export { RoomVisualizerConfigError } from "./config";

/**
 * Initialize the Room Visualizer widget inside the caller's page.
 *
 * Throws `RoomVisualizerConfigError` synchronously for any invalid
 * configuration (missing/malformed tenantId, unresolvable container)
 * — deliberately fails loudly rather than silently no-op'ing, so an
 * integrator immediately knows their setup is wrong instead of
 * wondering why the widget never appeared.
 */
export function init(config: RoomVisualizerConfig, callbacks?: WidgetCallbacks): WidgetHandle {
  const resolved = resolveConfig(config);
  return createWidget(resolved, callbacks);
}

const RoomVisualizer = { init, SDK_VERSION };
export default RoomVisualizer;
