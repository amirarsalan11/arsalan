import type { PublicMaterial } from "../types";

interface MaterialSelectorProps {
  materials: PublicMaterial[];
  selectedMaterialId: string | null;
  onSelect: (materialId: string | null) => void;
}

/**
 * Material selection grid, backed by GET /api/v1/public/tenants/{id}/materials
 * (active materials only, tenant-scoped, no storage details exposed —
 * see backend/app/schemas/public.py).
 *
 * Still a "placeholder" in the sense the approved plan means: this
 * only lets the user pick a materialId to carry into the render
 * request placeholder below. It has no rendering/preview capability
 * of its own — that's the AI pipeline's job, out of scope here.
 */
export function MaterialSelector({
  materials,
  selectedMaterialId,
  onSelect,
}: MaterialSelectorProps): JSX.Element {
  if (materials.length === 0) {
    return <p style={{ color: "#666" }}>No materials available yet.</p>;
  }

  return (
    <div
      role="radiogroup"
      aria-label="Select a material"
      style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}
    >
      {materials.map((material) => {
        const isSelected = material.id === selectedMaterialId;
        return (
          <button
            key={material.id}
            type="button"
            role="radio"
            aria-checked={isSelected}
            onClick={() => onSelect(isSelected ? null : material.id)}
            style={{
              padding: "0.5rem 0.75rem",
              borderRadius: "6px",
              border: isSelected ? "2px solid #333" : "1px solid #ccc",
              background: isSelected ? "#f0f0f0" : "#fff",
              cursor: "pointer",
            }}
          >
            <div style={{ fontWeight: 600 }}>{material.name}</div>
            <div style={{ fontSize: "0.75rem", color: "#777" }}>{material.category}</div>
          </button>
        );
      })}
    </div>
  );
}
