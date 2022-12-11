from functools import lru_cache
from pathlib import Path

from pydantic import BaseSettings, Field

from .make_reco import KionReco, KionRecoBM25

BASE_DIR = Path(__file__).resolve().parent


class Config(BaseSettings):
    class Config:
        case_sensitive = False
        env_file = str(BASE_DIR / ".env")


class LogConfig(Config):
    level: str = "INFO"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"

    class Config:
        case_sensitive = False
        fields = {
            "level": {
                "env": ["log_level"]
            },
        }


class ServiceConfig(Config):
    service_name: str = "reco_service"
    k_recs: int = 10
    # путь до данных дня поднятия в app.py
    items_path = Path.cwd().joinpath("service", "data", "kion_train",
                                     "kion_train",
                                     "interactions.csv")
    models = {
        "LightFM_0.078294": KionReco(Path.cwd().joinpath("service", "models",
                                                         "LightFM_0.078294.dill"),
                                     Path.cwd().joinpath("service",
                                                         "data",
                                                         "dataset_LightFM_0.078294.dill")),
        "BM25Recommender_0.085430": KionReco(
            Path.cwd().joinpath("service", "models",
                                "BM25Recommender_0.095432.dill"),
            Path.cwd().joinpath("service",
                                "data",
                                "dataset_BM25Recommender_0.095432.dill")),
        "userknn_BM25Recommender": KionRecoBM25(
            Path.cwd().joinpath("service", "models",
                                "userknn_BM25Recommender.dill"),
            Path.cwd().joinpath("service",
                                "data",
                                "dataset_userknn_BM25Recommender.dill"))}
    log_config: LogConfig
    secret_token: str = Field(None, env="SECRET_TOKEN")


@lru_cache()
def get_config() -> Config:
    return ServiceConfig(
        log_config=LogConfig(),
    )
