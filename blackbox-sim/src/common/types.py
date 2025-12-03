from dataclasses import dataclass
from typing import List, Optional

# Equivalent to struct TrainingConfig
@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    input_dim: int
    model_name: str

# Equivalent to struct LogSample
@dataclass
class LogSample:
    raw_text: str
    tokenized_vector: List[int]
    label: int  # 0 = Normal, 1 = Anomaly