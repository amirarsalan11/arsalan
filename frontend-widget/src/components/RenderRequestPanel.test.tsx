import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { RenderRequestPanel } from "./RenderRequestPanel";

describe("RenderRequestPanel", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("disables the Generate button when there is no image", () => {
    render(<RenderRequestPanel hasImage={false} selectedMaterialId={null} />);
    expect(screen.getByRole("button", { name: /generate/i })).toBeDisabled();
  });

  it("enables the Generate button once an image is present", () => {
    render(<RenderRequestPanel hasImage={true} selectedMaterialId={null} />);
    expect(screen.getByRole("button", { name: /generate/i })).toBeEnabled();
  });

  it("never calls the authenticated render-jobs endpoint", () => {
    // The load-bearing guarantee of this component: it must not call
    // the real, Bearer-authenticated POST /render-jobs endpoint, since
    // the widget has no secret key to authenticate with.
    //
    // Note: with vi.useFakeTimers() active, we deliberately do NOT use
    // `waitFor` here — its internal polling relies on real timers,
    // which are faked in this suite, so it would hang until Vitest's
    // own test timeout. Advancing the fake clock inside `act()` flushes
    // the resulting state update synchronously, so a direct assertion
    // immediately afterward is both correct and sufficient.
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    render(<RenderRequestPanel hasImage={true} selectedMaterialId="m1" />);
    fireEvent.click(screen.getByRole("button", { name: /generate/i }));

    act(() => {
      vi.advanceTimersByTime(1000);
    });

    expect(screen.getByText(/isn't available yet/i)).toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
    fetchSpy.mockRestore();
  });

  it("transitions to a loading state, then an honest unavailable message", () => {
    render(<RenderRequestPanel hasImage={true} selectedMaterialId={null} />);

    // fireEvent.click wraps dispatch in act() itself, so the resulting
    // setState("pending") is flushed to the DOM before this assertion
    // runs — unlike a raw `.click()` call, which gives no such guarantee
    // under React 18's batching and was the direct cause of this
    // assertion previously failing.
    fireEvent.click(screen.getByRole("button", { name: /generate/i }));
    expect(screen.getByText(/preparing your request/i)).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(1000);
    });

    expect(screen.getByText(/isn't available yet/i)).toBeInTheDocument();
  });
});
