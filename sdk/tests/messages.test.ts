/**
 * Tests for messages.ts — the widget/SDK message type guards.
 *
 * These specifically probe the "don't trust unless the discriminator
 * and shape both check out" property, since that's what stands
 * between this channel and a confused-deputy-style message injection.
 */

import { describe, expect, it } from "vitest";
import {
  isSdkMessage,
  isWidgetMessage,
  SDK_MESSAGE_SOURCE,
  WIDGET_MESSAGE_SOURCE,
} from "../src/messages";

describe("isWidgetMessage", () => {
  it("accepts a well-formed ready message", () => {
    expect(isWidgetMessage({ source: WIDGET_MESSAGE_SOURCE, type: "ready" })).toBe(true);
  });

  it("accepts a well-formed resize message", () => {
    expect(
      isWidgetMessage({ source: WIDGET_MESSAGE_SOURCE, type: "resize", height: 640 }),
    ).toBe(true);
  });

  it("rejects a resize message with a non-numeric height", () => {
    expect(
      isWidgetMessage({ source: WIDGET_MESSAGE_SOURCE, type: "resize", height: "640" }),
    ).toBe(false);
  });

  it("accepts a well-formed render-requested message", () => {
    expect(
      isWidgetMessage({
        source: WIDGET_MESSAGE_SOURCE,
        type: "render-requested",
        payload: { materialId: "abc-123" },
      }),
    ).toBe(true);
  });

  it("accepts render-requested with a null materialId", () => {
    expect(
      isWidgetMessage({
        source: WIDGET_MESSAGE_SOURCE,
        type: "render-requested",
        payload: { materialId: null },
      }),
    ).toBe(true);
  });

  it("rejects a message with the wrong source discriminator", () => {
    expect(isWidgetMessage({ source: "some-other-script", type: "ready" })).toBe(false);
  });

  it("rejects a message with an unknown type", () => {
    expect(isWidgetMessage({ source: WIDGET_MESSAGE_SOURCE, type: "totally-made-up" })).toBe(
      false,
    );
  });

  it("rejects non-object data entirely", () => {
    expect(isWidgetMessage("just a string")).toBe(false);
    expect(isWidgetMessage(null)).toBe(false);
    expect(isWidgetMessage(undefined)).toBe(false);
    expect(isWidgetMessage(42)).toBe(false);
  });

  it("rejects a message that merely resembles the shape but has no source field", () => {
    expect(isWidgetMessage({ type: "ready" })).toBe(false);
  });
});

describe("isSdkMessage", () => {
  it("accepts a well-formed config message", () => {
    expect(
      isSdkMessage({
        source: SDK_MESSAGE_SOURCE,
        type: "config",
        payload: { tenantId: "abc-123", apiBaseUrl: "https://api.example.com" },
      }),
    ).toBe(true);
  });

  it("rejects a config message missing apiBaseUrl", () => {
    expect(
      isSdkMessage({
        source: SDK_MESSAGE_SOURCE,
        type: "config",
        payload: { tenantId: "abc-123" },
      }),
    ).toBe(false);
  });

  it("rejects a message with the wrong source", () => {
    expect(
      isSdkMessage({
        source: "not-the-sdk",
        type: "config",
        payload: { tenantId: "abc-123", apiBaseUrl: "https://api.example.com" },
      }),
    ).toBe(false);
  });

  it("never treats a widget-sourced message as an SDK message", () => {
    expect(
      isSdkMessage({ source: WIDGET_MESSAGE_SOURCE, type: "ready" }),
    ).toBe(false);
  });
});
