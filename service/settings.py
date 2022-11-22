from pathlib import Path

from pydantic import BaseSettings


class Config(BaseSettings):
    class Config:
        case_sensitive = False


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


class ServiceTestConfig(Config):
    service_name: str = "reco_service"
    k_recs: int = 10
    items_path = Path.cwd().parent.parent.joinpath('service', 'data',
                                                   'kion_train', 'kion_train',
                                                   'interactions.csv')
    log_config: LogConfig
    model: str = 'first'


class ServiceConfig(Config):
    service_name: str = "reco_service"
    k_recs: int = 10

    items_path = Path.cwd().joinpath('service', 'data', 'kion_train',
                                     'kion_train',
                                     'interactions.csv')
    log_config: LogConfig
    model: str = 'first'


def get_config() -> Config:
    return ServiceConfig(
        log_config=LogConfig(),
    )
