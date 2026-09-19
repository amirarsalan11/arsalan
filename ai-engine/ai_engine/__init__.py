"""
AI Engine package.

Independent from the FastAPI backend, per architecture lock:

    "The AI Engine is independent from FastAPI."

Pipeline stages (each isolated, own package):
    preprocessing -> segmentation -> mask_processing -> geometry
    -> material -> perspective -> lighting -> compositing

Milestone 1 scope: package/module structure only. No pipeline
implementation, no model loading, no OpenCV/PyTorch usage yet.
"""
