from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING

from config_schemas import training_params_schema

@dataclass
class Config:
    training_params: training_params_schema.TrainingParamsConfig = MISSING

def setup_config() -> None:
    training_params_schema.setup_config()

    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
    