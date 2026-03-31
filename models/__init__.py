"""Model architectures for the multimodal training project.

This package currently provides two variants:
- archA: Transformer-based fusion with positional embeddings
- archB: Late fusion with MLP classifier

You can import a specific architecture, for example:
    from models.archA import CLIPElectraFusion, EarlyStopping
or:
    from models.archB import CLIPElectraFusion, EarlyStopping
"""

__all__ = [
    "archA",
    "archB",
]
