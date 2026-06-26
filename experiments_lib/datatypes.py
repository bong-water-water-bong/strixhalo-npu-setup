from dataclasses import dataclass


@dataclass(frozen=True)
class DataType:
    name: str
    bit_width: int
    element_size: int  # bytes

    def __str__(self):
        return self.name


BFP16 = DataType("bfp16", 16, 2)
BF16 = DataType("bf16", 16, 2)
FP32 = DataType("fp32", 32, 4)


def from_string(s: str) -> DataType:
    mapping = {"bfp16": BFP16, "bf16": BF16, "fp32": FP32}
    return mapping[s.lower()]