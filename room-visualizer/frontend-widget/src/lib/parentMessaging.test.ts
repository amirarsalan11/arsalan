import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { listenForSdkConfig, notifyReady, notifyRenderRequested } from "./parentMessaging";

const PARENT_ORIGIN = "https://shop.example.com";

function setEmbedded(embedded: boolean): void {
  if (embedded) {
    // jsdom's window.parent === window by default (not embedded);
    // simulate an embedded frame by overriding it.
    Object.defineProperty(window, "parent", {
      value: { postMessage: vi.fn() },
      configurable: true,
    });
  } else {
    Object.defineProperty(window, "parent", { value: window, configurable: true });
  }
}

function setReferrer(url: string): void {
  Object.defineProperty(document, "referrer", { value: url, configurable: true });
}

describe("parentMessaging", () => {
  afterEach(() => {
    setEmbedded(false);
    setReferrer("");
    vi.restoreAllMocks();
  });

  describe("sendToParent (via notifyReady)", () => {
    it("does nothing when the widget is not embedded", () => {
      setEmbedded(false);
      // window.parent === window here, so postMessage on the real
      // window would actually dispatch a message event to ourselves;
      // spy on it to confirm it's never called.
      const postMessageSpy = vi.spyOn(window, "postMessage");
      notifyReady();
      expect(postMessageSpy).not.toHaveBeenCalled();
    });

    it("does nothing when no parent origin can be determined", () => {
      setEmbedded(true);
      setReferrer("");
      notifyReady();
      expect((window.parent as unknown as { postMessage: ReturnType<typeof vi.fn> }).postMessage).not.toHaveBeenCalled();
    });

    it("posts to the parent with the exact referrer origin as target, never '*'", () => {
      setEmbedded(true);
      setReferrer(`${PARENT_ORIGIN}/some/page`);

      notifyReady();

      const postMessage = (window.parent as unknown as { postMessage: ReturnType<typeof vi.fn> })
        .postMessage;
      expect(postMessage).toHaveBeenCalledWith(
        { source: "room-visualizer-widget", type: "ready" },
        PARENT_ORIGIN,
      );
    });

    it("never sends a message with a secret/API-key-shaped field", () => {
      setEmbedded(true);
      setReferrer(PARENT_ORIGIN);

      notifyRenderRequested("m1");

      const postMessage = (window.parent as unknown as { postMessage: ReturnType<typeof vi.fn> })
        .postMessage;
      const sentMessage = postMessage.mock.calls[0]?.[0];
      const serialized = JSON.stringify(sentMessage);
      expect(serialized).not.toMatch(/rv_live_/);
      expect(serialized.toLowerCase()).not.toContain("apikey");
      expect(serialized.toLowerCase()).not.toContain("api_key");
    });
  });

  describe("listenForSdkConfig", () => {
    beforeEach(() => {
      setEmbedded(true);
      setReferrer(`${PARENT_ORIGIN}/page`);
    });

    it("invokes the callback for a well-formed config message from the correct origin", () => {
      const onConfig = vi.fn();
      listenForSdkConfig(onConfig);

      window.dispatchEvent(
        new MessageEvent("message", {
          origin: PARENT_ORIGIN,
          source: window.parent as unknown as Window,
          data: {
            source: "room-visualizer-sdk",
            type: "config",
            payload: { tenantId: "t1", apiBaseUrl: "https://api.example.com" },
          },
        }),
      );

      expect(onConfig).toHaveBeenCalledWith({
        tenantId: "t1",
        apiBaseUrl: "https://api.example.com",
      });
    });

    it("ignores a message from a mismatched origin", () => {
      const onConfig = vi.fn();
      listenForSdkConfig(onConfig);

      window.dispatchEvent(
        new MessageEvent("message", {
          origin: "https://attacker.example.com",
          source: window.parent as unknown as Window,
          data: {
            source: "room-visualizer-sdk",
            type: "config",
            payload: { tenantId: "t1", apiBaseUrl: "https://api.example.com" },
          },
        }),
      );

      expect(onConfig).not.toHaveBeenCalled();
    });

    it("ignores a message with the wrong source discriminator, even from the right origin", () => {
      const onConfig = vi.fn();
      listenForSdkConfig(onConfig);

      window.dispatchEvent(
        new MessageEvent("message", {
          origin: PARENT_ORIGIN,
          source: window.parent as unknown as Window,
          data: { source: "not-the-sdk", type: "config", payload: {} },
        }),
      );

      expect(onConfig).not.toHaveBeenCalled();
    });

    it("returns an unsubscribe function that stops future callbacks", () => {
      const onConfig = vi.fn();
      const unsubscribe = listenForSdkConfig(onConfig);
      unsubscribe();

      window.dispatchEvent(
        new MessageEvent("message", {
          origin: PARENT_ORIGIN,
          source: window.parent as unknown as Window,
          data: {
            source: "room-visualizer-sdk",
            type: "config",
            payload: { tenantId: "t1", apiBaseUrl: "https://api.example.com" },
          },
        }),
      );

      expect(onConfig).not.toHaveBeenCalled();
    });
  });
});
