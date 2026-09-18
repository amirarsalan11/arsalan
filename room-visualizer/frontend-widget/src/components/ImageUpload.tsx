import { useCallback, useEffect, useRef, useState } from "react";

interface ImageUploadProps {
  onImageSelected: (file: File | null) => void;
}

/**
 * Room image upload UI.
 *
 * SDK + Widget milestone scope: client-side file selection and local
 * preview ONLY, via `URL.createObjectURL`. Nothing is uploaded to any
 * backend or storage provider — there is no storage integration yet
 * (explicitly out of scope), and this component has no knowledge of
 * one. `onImageSelected` just hands the raw `File` up to the parent
 * screen for local state tracking.
 */
export function ImageUpload({ onImageSelected }: ImageUploadProps): JSX.Element {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Revoke the previous object URL whenever it changes or on unmount,
    // to avoid leaking memory across repeated selections.
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFile = useCallback(
    (file: File | null) => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }

      if (!file) {
        setPreviewUrl(null);
        setFileName(null);
        onImageSelected(null);
        return;
      }

      setPreviewUrl(URL.createObjectURL(file));
      setFileName(file.name);
      onImageSelected(file);
    },
    [onImageSelected, previewUrl],
  );

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>): void => {
    handleFile(event.target.files?.[0] ?? null);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>): void => {
    event.preventDefault();
    handleFile(event.dataTransfer.files?.[0] ?? null);
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>): void => {
    event.preventDefault();
  };

  return (
    <div>
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        style={{
          border: "2px dashed #ccc",
          borderRadius: "8px",
          padding: "1.5rem",
          textAlign: "center",
          cursor: "pointer",
        }}
      >
        {previewUrl ? (
          <img
            src={previewUrl}
            alt={fileName ?? "Selected room"}
            style={{ maxWidth: "100%", maxHeight: "240px", borderRadius: "4px" }}
          />
        ) : (
          <p>Click or drag a room photo here</p>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={handleInputChange}
          style={{ display: "none" }}
        />
      </div>
      {fileName && <p style={{ fontSize: "0.85rem", color: "#666" }}>{fileName}</p>}
    </div>
  );
}
