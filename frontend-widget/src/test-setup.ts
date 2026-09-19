import "@testing-library/jest-dom";

// jsdom does not implement URL.createObjectURL/revokeObjectURL, but
// ImageUpload.tsx relies on them for local-only preview rendering.
// Stub them so component tests can run without a real browser.
if (typeof URL.createObjectURL !== "function") {
  URL.createObjectURL = (): string => "blob:mock-object-url";
}
if (typeof URL.revokeObjectURL !== "function") {
  URL.revokeObjectURL = (): void => {};
}
