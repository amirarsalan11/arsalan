/**
 * Tests for config.ts — resolveConfig() validation and normalization.
 *
 * Run with: npm test (vitest + jsdom, see sdk/package.json).
 */

import { describe, expect, it, beforeEach } from "vitest";
import { resolveConfig, RoomVisualizerConfigError } from "../src/config";

const VALID_TENANT_ID = "a1b2c3d4-e5f6-47a8-89b0-1234567890ab";

describe("resolveConfig", () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="target"></div>';
  });

  it("resolves a valid config with a CSS selector container", () => {
    const resolved = resolveConfig({
      tenantId: VALID_TENANT_ID,
      container: "#target",
    });

    expect(resolved.tenantId).toBe(VALID_TENANT_ID);
    expect(resolved.container).toBeInstanceOf(HTMLElement);
    expect(resolved.widgetOrigin).toBe("https://widget.roomvisualizer.com");
    expect(resolved.apiBaseUrl).toBe("https://api.roomvisualizer.com");
  });

  it("accepts a real HTMLElement as the container", () => {
    const element = document.getElementById("target")!;
    const resolved = resolveConfig({ tenantId: VALID_TENANT_ID, container: element });
    expect(resolved.container).toBe(element);
  });

  it("throws when tenantId is missing", () => {
    expect(() =>
      // @ts-expect-error intentionally omitting a required field
      resolveConfig({ container: "#target" }),
    ).toThrow(RoomVisualizerConfigError);
  });

  it("throws when tenantId is not a UUID", () => {
    expect(() =>
      resolveConfig({ tenantId: "not-a-uuid", container: "#target" }),
    ).toThrow(RoomVisualizerConfigError);
  });

  it("throws when tenantId looks like an API key rather than a UUID", () => {
    // Guards against the exact mistake this milestone must prevent:
    // someone passing a secret key where a public tenantId belongs.
    expect(() =>
      resolveConfig({ tenantId: "rv_live_abc123", container: "#target" }),
    ).toThrow(RoomVisualizerConfigError);
  });

  it("throws when the container selector matches nothing", () => {
    expect(() =>
      resolveConfig({ tenantId: VALID_TENANT_ID, container: "#does-not-exist" }),
    ).toThrow(RoomVisualizerConfigError);
  });

  it("respects a caller-supplied widgetOrigin override", () => {
    const resolved = resolveConfig({
      tenantId: VALID_TENANT_ID,
      container: "#target",
      widgetOrigin: "http://localhost:5173/",
    });
    // Trailing slash stripped so origin comparisons stay exact.
    expect(resolved.widgetOrigin).toBe("http://localhost:5173");
  });

  it("respects a caller-supplied apiBaseUrl override", () => {
    const resolved = resolveConfig({
      tenantId: VALID_TENANT_ID,
      container: "#target",
      apiBaseUrl: "http://localhost:8000",
    });
    expect(resolved.apiBaseUrl).toBe("http://localhost:8000");
  });
});
