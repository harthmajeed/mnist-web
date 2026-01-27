from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class TrainingParamsConfig:
    _target_: str = MISSING

@dataclass
class Default(TrainingParamsConfig):
    _target_: str = "default"
    batch_size = MISSING
    epochs = MISSING

def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="training_params", name="default", node=Default)
    