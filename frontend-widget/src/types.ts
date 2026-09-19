/**
 * Shared types for the widget (EmbedApp and its components).
 */

export interface PublicTenantConfig {
  tenant_id: string;
  name: string;
  is_active: boolean;
}

export interface PublicMaterial {
  id: string;
  name: string;
  category: string;
}

export type LoadState<T> =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "success"; data: T };
