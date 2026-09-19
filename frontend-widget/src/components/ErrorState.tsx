interface ErrorStateProps {
  message: string;
}

export function ErrorState({ message }: ErrorStateProps): JSX.Element {
  return (
    <div
      role="alert"
      style={{
        padding: "1rem",
        textAlign: "center",
        color: "#a33",
        border: "1px solid #eab",
        borderRadius: "6px",
        background: "#fdf2f2",
      }}
    >
      {message}
    </div>
  );
}
