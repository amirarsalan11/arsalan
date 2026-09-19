import { describe, expect, it, vi, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { EmbedApp } from "./EmbedApp";

function setUrl(search: string): void {
  window.history.pushState({}, "", `/embed${search}`);
}

const TENANT_ID = "a1b2c3d4-e5f6-47a8-89b0-1234567890ab";

describe("EmbedApp", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    setUrl("");
  });

  it("shows an error when no tenantId is present in the URL", () => {
    setUrl("");
    render(<EmbedApp />);
    expect(screen.getByRole("alert")).toHaveTextContent(/missing tenant configuration/i);
  });

  it("loads tenant config and materials from the public endpoints", async () => {
    setUrl(`?tenantId=${TENANT_ID}`);

    vi.spyOn(globalThis, "fetch").mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/config")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({ tenant_id: TENANT_ID, name: "Acme Corp", is_active: true }),
            { status: 200 },
          ),
        );
      }
      if (url.includes("/materials")) {
        return Promise.resolve(
          new Response(
            JSON.stringify([{ id: "m1", name: "Oak Hardwood", category: "hardwood" }]),
            { status: 200 },
          ),
        );
      }
      return Promise.reject(new Error(`Unexpected fetch: ${url}`));
    });

    render(<EmbedApp />);

    await waitFor(() => expect(screen.getByText("Acme Corp")).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText("Oak Hardwood")).toBeInTheDocument());
  });

  it("shows an error state when the tenant config request fails", async () => {
    setUrl(`?tenantId=${TENANT_ID}`);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("Forbidden", { status: 403 }));

    render(<EmbedApp />);

    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
  });

  it("never sends an Authorization header while loading", async () => {
    setUrl(`?tenantId=${TENANT_ID}`);
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify([]), { status: 200 }));

    render(<EmbedApp />);
    await waitFor(() => expect(fetchSpy).toHaveBeenCalled());

    for (const call of fetchSpy.mock.calls) {
      const options = call[1] as RequestInit | undefined;
      const headers = options?.headers as Record<string, string> | undefined;
      const hasAuthHeader = Object.keys(headers ?? {}).some(
        (key) => key.toLowerCase() === "authorization",
      );
      expect(hasAuthHeader).toBe(false);
    }
  });
});
