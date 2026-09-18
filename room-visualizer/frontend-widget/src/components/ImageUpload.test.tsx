import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ImageUpload } from "./ImageUpload";

function makeImageFile(name = "room.png"): File {
  return new File(["fake-image-bytes"], name, { type: "image/png" });
}

describe("ImageUpload", () => {
  it("shows a prompt before any file is selected", () => {
    render(<ImageUpload onImageSelected={vi.fn()} />);
    expect(screen.getByText(/click or drag a room photo/i)).toBeInTheDocument();
  });

  it("calls onImageSelected with the chosen file", () => {
    const onImageSelected = vi.fn();
    render(<ImageUpload onImageSelected={onImageSelected} />);

    const input = screen.getByRole("button").querySelector("input")!;
    const file = makeImageFile();
    fireEvent.change(input, { target: { files: [file] } });

    expect(onImageSelected).toHaveBeenCalledWith(file);
  });

  it("shows the selected file name after selection", () => {
    render(<ImageUpload onImageSelected={vi.fn()} />);

    const input = screen.getByRole("button").querySelector("input")!;
    fireEvent.change(input, { target: { files: [makeImageFile("my-room.jpg")] } });

    expect(screen.getByText("my-room.jpg")).toBeInTheDocument();
  });

  it("never attempts any network request on selection", () => {
    // Guard against a regression that would wire this up to a
    // storage/upload backend before that milestone exists.
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    render(<ImageUpload onImageSelected={vi.fn()} />);

    const input = screen.getByRole("button").querySelector("input")!;
    fireEvent.change(input, { target: { files: [makeImageFile()] } });

    expect(fetchSpy).not.toHaveBeenCalled();
    fetchSpy.mockRestore();
  });
});
