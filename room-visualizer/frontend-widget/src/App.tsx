/**
 * Root widget component for any route other than `/embed`.
 *
 * SDK + Widget milestone: the real widget screen now lives in
 * EmbedApp.tsx, mounted at `/embed` (see main.tsx) — that's what
 * actually runs inside the SDK's iframe. This component remains a
 * simple landing stub for the case where someone opens the widget's
 * bare URL directly, outside of any SDK/iframe context.
 */

export default function App(): JSX.Element {
  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: "1rem" }}>
      <p>
        This is the Room Visualizer widget host. It's designed to be embedded via
        the Room Visualizer SDK, not opened directly.
      </p>
    </div>
  );
}
