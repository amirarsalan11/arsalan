interface LoadingStateProps {
  label?: string;
}

export function LoadingState({ label = "Loading…" }: LoadingStateProps): JSX.Element {
  return (
    <div role="status" style={{ padding: "1rem", textAlign: "center", color: "#555" }}>
      {label}
    </div>
  );
}
