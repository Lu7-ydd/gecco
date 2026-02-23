from dataclasses import dataclass


@dataclass
class GeCCoConfig:
    phi_threshold: float = 0.3
    fdr_threshold: float = 0.05
    min_module_size: int = 2
    max_depth: int = 5
    binarize_quantile: float = 0.50
    random_state: int = 42
