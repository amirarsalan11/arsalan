/**
 * Backend communication layer for the widget.
 *
 * Every fetch to the backend goes through this module — no component
 * calls `fetch` directly — mirroring the layering discipline used
 * elsewhere in this project.
 *
 * Security note: this client NEVER attaches an `Authorization` header,
 * because it has no secret to attach. It calls only the public,
 * Origin-validated endpoints under `/api/v1/public/...`. If a future
 * change to this file starts sending any kind of API key, that is
 * itself a sign the widget/backend boundary has been violated — the
 * widget must never hold a tenant's secret key.
 */

import type { PublicMaterial, PublicTenantConfig } from "./types";

export class BackendRequestError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "BackendRequestError";
  }
}

export class BackendClient {
  constructor(
    private readonly apiBaseUrl: string,
    private readonly tenantId: string,
  ) {}

  async getTenantConfig(): Promise<PublicTenantConfig> {
    return this.getJson<PublicTenantConfig>(
      `/api/v1/public/tenants/${this.tenantId}/config`,
    );
  }

  async listMaterials(): Promise<PublicMaterial[]> {
    return this.getJson<PublicMaterial[]>(
      `/api/v1/public/tenants/${this.tenantId}/materials`,
    );
  }

  private async getJson<T>(path: string): Promise<T> {
    const url = `${this.apiBaseUrl}${path}`;

    let response: Response;
    try {
      response = await fetch(url, {
        method: "GET",
        // Deliberately no Authorization header — see module docstring.
        headers: { Accept: "application/json" },
      });
    } catch (cause) {
      throw new BackendRequestError(
        "Could not reach the Room Visualizer backend. Check your network connection.",
      );
    }

    if (!response.ok) {
      throw new BackendRequestError(
        `Request to ${path} failed with status ${response.status}.`,
        response.status,
      );
    }

    return (await response.json()) as T;
  }
}
