import { useEffect, useState } from "react";
import { BackendClient } from "./lib/backendClient";
import { listenForSdkConfig, notifyReady } from "./lib/parentMessaging";
import { ImageUpload } from "./components/ImageUpload";
import { MaterialSelector } from "./components/MaterialSelector";
import { RenderRequestPanel } from "./components/RenderRequestPanel";
import { LoadingState } from "./components/LoadingState";
import { ErrorState } from "./components/ErrorState";
import type { LoadState, PublicMaterial, PublicTenantConfig } from "./types";

const DEFAULT_API_BASE_URL = "https://api.roomvisualizer.com";

/**
 * The real widget screen, mounted at the `/embed` route (see
 * main.tsx). This is what actually runs inside the SDK's iframe.
 *
 * Identity/config resolution order:
 * 1. Read `tenantId` from the URL query string (the SDK always puts
 *    it there when constructing the iframe src — see sdk/src/iframe.ts).
 * 2. Also listen for the SDK's `config` postMessage, which carries the
 *    same tenantId plus `apiBaseUrl`. Whichever arrives, the widget
 *    uses it; the URL param lets the widget start fetching immediately
 *    without waiting on the message round-trip, while the message
 *    provides `apiBaseUrl` for non-default deployments.
 *
 * Never expects, reads, or stores anything resembling a secret API
 * key from either source — see backendClient.ts.
 */
export function EmbedApp(): JSX.Element {
  const [tenantId, setTenantId] = useState<string | null>(() => readTenantIdFromUrl());
  const [apiBaseUrl, setApiBaseUrl] = useState<string>(DEFAULT_API_BASE_URL);

  const [configState, setConfigState] = useState<LoadState<PublicTenantConfig> | null>(null);
  const [materialsState, setMaterialsState] = useState<LoadState<PublicMaterial[]> | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedMaterialId, setSelectedMaterialId] = useState<string | null>(null);

  // Announce readiness to the parent SDK as soon as we've mounted, and
  // listen for the SDK's config message in case it supplies a
  // non-default apiBaseUrl (e.g. local/staging development).
  useEffect(() => {
    notifyReady();
    const unsubscribe = listenForSdkConfig((payload) => {
      setTenantId(payload.tenantId);
      setApiBaseUrl(payload.apiBaseUrl);
    });
    return unsubscribe;
  }, []);

  useEffect(() => {
    if (!tenantId) {
      return;
    }

    const client = new BackendClient(apiBaseUrl, tenantId);

    setConfigState({ status: "loading" });
    client
      .getTenantConfig()
      .then((data) => setConfigState({ status: "success", data }))
      .catch((error: unknown) =>
        setConfigState({ status: "error", message: describeError(error) }),
      );

    setMaterialsState({ status: "loading" });
    client
      .listMaterials()
      .then((data) => setMaterialsState({ status: "success", data }))
      .catch((error: unknown) =>
        setMaterialsState({ status: "error", message: describeError(error) }),
      );
  }, [tenantId, apiBaseUrl]);

  if (!tenantId) {
    return (
      <ErrorState message="Missing tenant configuration. This widget must be embedded via the Room Visualizer SDK." />
    );
  }

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: "1rem", maxWidth: "480px" }}>
      {configState?.status === "loading" && <LoadingState label="Loading…" />}
      {configState?.status === "error" && <ErrorState message={configState.message} />}
      {configState?.status === "success" && !configState.data.is_active && (
        <ErrorState message="This Room Visualizer account is currently inactive." />
      )}

      {configState?.status === "success" && configState.data.is_active && (
        <>
          <h2 style={{ marginTop: 0 }}>{configState.data.name}</h2>

          <section style={{ marginBottom: "1rem" }}>
            <ImageUpload onImageSelected={setSelectedFile} />
          </section>

          <section style={{ marginBottom: "1rem" }}>
            {materialsState?.status === "loading" && <LoadingState label="Loading materials…" />}
            {materialsState?.status === "error" && <ErrorState message={materialsState.message} />}
            {materialsState?.status === "success" && (
              <MaterialSelector
                materials={materialsState.data}
                selectedMaterialId={selectedMaterialId}
                onSelect={setSelectedMaterialId}
              />
            )}
          </section>

          <RenderRequestPanel hasImage={selectedFile !== null} selectedMaterialId={selectedMaterialId} />
        </>
      )}
    </div>
  );
}

function readTenantIdFromUrl(): string | null {
  const params = new URLSearchParams(window.location.search);
  return params.get("tenantId");
}

function describeError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return "Something went wrong loading this widget.";
}
