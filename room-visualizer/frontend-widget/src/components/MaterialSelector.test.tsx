import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MaterialSelector } from "./MaterialSelector";

const MATERIALS = [
  { id: "m1", name: "Oak Hardwood", category: "hardwood" },
  { id: "m2", name: "Slate Tile", category: "tile" },
];

describe("MaterialSelector", () => {
  it("shows a fallback message when there are no materials", () => {
    render(
      <MaterialSelector materials={[]} selectedMaterialId={null} onSelect={vi.fn()} />,
    );
    expect(screen.getByText(/no materials available/i)).toBeInTheDocument();
  });

  it("renders every provided material", () => {
    render(
      <MaterialSelector materials={MATERIALS} selectedMaterialId={null} onSelect={vi.fn()} />,
    );
    expect(screen.getByText("Oak Hardwood")).toBeInTheDocument();
    expect(screen.getByText("Slate Tile")).toBeInTheDocument();
  });

  it("never renders any storage/texture URL for a material", () => {
    render(
      <MaterialSelector materials={MATERIALS} selectedMaterialId={null} onSelect={vi.fn()} />,
    );
    // The public materials type has no texture_url field at all, but
    // this also guards against a future accidental regression that
    // widens the type and starts rendering it.
    expect(screen.queryByText(/https?:\/\//)).not.toBeInTheDocument();
  });

  it("calls onSelect with the material id when an unselected option is clicked", () => {
    const onSelect = vi.fn();
    render(
      <MaterialSelector materials={MATERIALS} selectedMaterialId={null} onSelect={onSelect} />,
    );

    screen.getByText("Oak Hardwood").closest("button")!.click();
    expect(onSelect).toHaveBeenCalledWith("m1");
  });

  it("calls onSelect with null when the already-selected option is clicked again", () => {
    const onSelect = vi.fn();
    render(
      <MaterialSelector materials={MATERIALS} selectedMaterialId="m1" onSelect={onSelect} />,
    );

    screen.getByText("Oak Hardwood").closest("button")!.click();
    expect(onSelect).toHaveBeenCalledWith(null);
  });

  it("marks the selected material as checked", () => {
    render(
      <MaterialSelector materials={MATERIALS} selectedMaterialId="m2" onSelect={vi.fn()} />,
    );

    const selectedButton = screen.getByText("Slate Tile").closest("button")!;
    expect(selectedButton).toHaveAttribute("aria-checked", "true");
  });
});
