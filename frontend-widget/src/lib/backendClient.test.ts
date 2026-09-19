import { describe, expect, it, vi, afterEach } from "vitest";
import { BackendClient, BackendRequestError } from "./backendClient";

const TENANT_ID = "a1b2c3d4-e5f6-47a8-89b0-1234567890ab";
const API_BASE_URL = "https://api.example.com";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("BackendClient", () => {
  it("fetches tenant config from the correct public path", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ tenant_id: TENANT_ID, name: "Acme", is_active: true }), {
        status: 200,
      }),
    );

    const client = new BackendClient(API_BASE_URL, TENANT_ID);
    const config = await client.getTenantConfig();

    expect(fetchSpy).toHaveBeenCalledWith(
      `${API_BASE_URL}/api/v1/public/tenants/${TENANT_ID}/config`,
      expect.objectContaining({ method: "GET" }),
    );
    expect(config.name).toBe("Acme");
  });

  it("never attaches an Authorization header to any request", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify([]), { status: 200 }));

    const client = new BackendClient(API_BASE_URL, TENANT_ID);
    await client.listMaterials();

    const [, options] = fetchSpy.mock.calls[0]!;
    const headers = options?.headers as Record<string, string> | undefined;
    expect(headers).toBeDefined();
    expect(Object.keys(headers ?? {}).some((key) => key.toLowerCase() === "authorization")).toBe(
      false,
    );
  });

  it("throws BackendRequestError on a non-OK response", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("Forbidden", { status: 403 }));

    const client = new BackendClient(API_BASE_URL, TENANT_ID);
    await expect(client.getTenantConfig()).rejects.toThrow(BackendRequestError);
  });

  it("throws BackendRequestError when fetch itself rejects (network failure)", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("network down"));

    const client = new BackendClient(API_BASE_URL, TENANT_ID);
    await expect(client.listMaterials()).rejects.toThrow(BackendRequestError);
  });

  it("fetches the materials list from the correct public path", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify([{ id: "m1", name: "Oak", category: "hardwood" }]), {
        status: 200,
      }),
    );

    const client = new BackendClient(API_BASE_URL, TENANT_ID);
    const materials = await client.listMaterials();

    expect(fetchSpy).toHaveBeenCalledWith(
      `${API_BASE_URL}/api/v1/public/tenants/${TENANT_ID}/materials`,
      expect.anything(),
    );
    expect(materials).toHaveLength(1);
  });
});
