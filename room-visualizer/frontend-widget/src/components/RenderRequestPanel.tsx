import { useState } from "react";
import { notifyRenderRequested } from "../lib/parentMessaging";
import { LoadingState } from "./LoadingState";

interface RenderRequestPanelProps {
  hasImage: boolean;
  selectedMaterialId: string | null;
}

type PanelState = "idle" | "pending" | "unavailable";

/**
 * Render request UI — deliberately a PLACEHOLDER, per the approved
 * scope. Clicking "Generate" does NOT call the real, authenticated
 * `POST /api/v1/render-jobs` endpoint: that endpoint requires a secret
 * Bearer API key, which the widget correctly never has (see the
 * Authentication and SDK+Widget milestone security discussions). It
 * also does not implement any AI rendering, Celery dispatch, or
 * storage upload — all explicitly out of scope for this milestone.
 *
 * What it DOES do, for real: transitions through a loading state and
 * notifies the parent SDK (via postMessage) that a render was
 * requested, carrying only the selected materialId — useful signal
 * for an integrator's own analytics even before real rendering exists.
 */
export function RenderRequestPanel({
  hasImage,
  selectedMaterialId,
}: RenderRequestPanelProps): JSX.Element {
  const [state, setState] = useState<PanelState>("idle");

  const handleGenerateClick = (): void => {
    setState("pending");
    notifyRenderRequested(selectedMaterialId);

    // Simulated delay so the loading state is visibly real, then land
    // on the honest "not implemented yet" outcome — no fake success,
    // no fabricated result image.
    window.setTimeout(() => setState("unavailable"), 600);
  };

  if (state === "pending") {
    return <LoadingState label="Preparing your request…" />;
  }

  if (state === "unavailable") {
    return (
      <div role="status" style={{ padding: "1rem", color: "#666" }}>
        Rendering isn't available yet — this is a preview of the Room
        Visualizer flow. Check back soon for real AI-generated results.
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={handleGenerateClick}
      disabled={!hasImage}
      style={{
        padding: "0.6rem 1.2rem",
        borderRadius: "6px",
        border: "none",
        background: hasImage ? "#333" : "#ccc",
        color: "#fff",
        cursor: hasImage ? "pointer" : "not-allowed",
      }}
    >
      Generate
    </button>
  );
}
