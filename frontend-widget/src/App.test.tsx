import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import App from "./App";

// The non-embed landing stub. Real interaction tests for the actual
// widget screen live in EmbedApp.test.tsx and the component-level
// test files under src/components/.
describe("App shell (non-embed landing stub)", () => {
  it("renders without crashing", () => {
    render(<App />);
    expect(screen.getByText(/embedded via/i)).toBeInTheDocument();
  });
});
