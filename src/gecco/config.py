from dataclasses import dataclass


@dataclass
class GeCCoConfig:
    phi_threshold: float = 0.0
    fdr_threshold: float = 0.05
    min_module_size: int = 1
    max_depth: int = 5
    binarize_quantile: float | None = None
    random_state: int = 42
    relationship_method: str = "binary"
    test: bool = False
    max_pair_violation_fraction: float = 0.0
    pp_enabled: bool = False
    pp_n_depth1: int = 3
    pp_n_depth2: int | list[int] = 3
    pp_branching: tuple[int, ...] | None = None
    pp_pos_pair_frac: float = 0.8
    pp_max_assign_depth: int | None = None

    def __post_init__(self):
        if not 0 <= self.max_pair_violation_fraction < 0.5:
            raise ValueError("max_pair_violation_fraction must be in [0, 0.5)")
        if self.phi_threshold < 0:
            raise ValueError("phi_threshold must be nonnegative")
        if self.max_depth < 1 or self.min_module_size < 1:
            raise ValueError("max_depth and min_module_size must be at least 1")
