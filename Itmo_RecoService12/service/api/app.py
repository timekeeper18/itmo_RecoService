import asyncio
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any, Dict

import pandas as pd
import uvloop
from fastapi import FastAPI

from .exception_handlers import add_exception_handlers
from .middlewares import add_middlewares
from .views import add_views
from ..log import app_logger, setup_logging
from ..make_reco import KionReco
from ..settings import ServiceConfig

__all__ = ("create_app",)


def setup_asyncio(thread_name_prefix: str) -> None:
    uvloop.install()

    loop = asyncio.get_event_loop()

    executor = ThreadPoolExecutor(thread_name_prefix=thread_name_prefix)
    loop.set_default_executor(executor)

    def handler(_, context: Dict[str, Any]) -> None:
        message = "Caught asyncio exception: {message}".format_map(context)
        app_logger.warning(message)

    loop.set_exception_handler(handler)


def create_app(config: ServiceConfig) -> FastAPI:
    setup_logging(config)
    setup_asyncio(thread_name_prefix=config.service_name)
    app = FastAPI(debug=False)
    app.state.k_recs = config.k_recs
    app.state.items_path = config.items_path
    # Импортируем из конфига наименование модели
    app.state.models = config.models
    # поднимаем и подготавливаем данные
    a = pd.read_csv(config.items_path)[["user_id", "item_id"]]
    app.state.item_list = list(a["item_id"].unique())
    app.state.items = a.groupby("user_id").agg(
        {"item_id": lambda x: sorted(list(x))}).reset_index()

    # инициализируем класс с рекомендациями
    # app.state.lightfm_0077652 = KionReco(config.lightfm_path,
    #                                      config.dataset_path)

    add_views(app)
    add_middlewares(app)
    add_exception_handlers(app)

    return app
