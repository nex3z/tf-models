import dataclasses


@dataclasses.dataclass()
class BlockConfig:
    kernel_size: int
    filters_in: int
    filters_out: int
    strides: int = 1
    expand_ratio: int = 1
    activation: str = 'swish'
    se_ratio: float = 0.0
    drop_rate: float = 0.0
    id_skip: bool = True,
    repeats: int = 1


DEFAULT_BLOCKS_ARGS = [
    BlockConfig(
        kernel_size=3,
        repeats=1,
        filters_in=32,
        filters_out=16,
        expand_ratio=1,
        id_skip=True,
        strides=1,
        se_ratio=0.25
    ),
    BlockConfig(
        kernel_size=3,
        repeats=2,
        filters_in=16,
        filters_out=24,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25
    ),
    BlockConfig(
        kernel_size=5,
        repeats=2,
        filters_in=25,
        filters_out=40,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25
    ),
    BlockConfig(
        kernel_size=3,
        repeats=3,
        filters_in=40,
        filters_out=80,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25
    ),
    BlockConfig(
        kernel_size=5,
        repeats=3,
        filters_in=80,
        filters_out=112,
        expand_ratio=6,
        id_skip=True,
        strides=1,
        se_ratio=0.25
    ),
    BlockConfig(
        kernel_size=5,
        repeats=4,
        filters_in=112,
        filters_out=192,
        expand_ratio=6,
        id_skip=True,
        strides=2,
        se_ratio=0.25
    ),
    BlockConfig(
        kernel_size=3,
        repeats=1,
        filters_in=192,
        filters_out=320,
        expand_ratio=6,
        id_skip=True,
        strides=1,
        se_ratio=0.25
    ),
]
