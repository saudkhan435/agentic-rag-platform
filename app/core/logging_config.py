import logging.config
import os


def configure_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,

            "formatters": {
                "default": {
                    "format": (
                        "%(asctime)s | "
                        "%(levelname)s | "
                        "%(name)s | "
                        "%(message)s"
                    )
                }
            },

            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                }
            },

            "loggers": {
                "app": {
                    "handlers": ["console"],
                    "level": log_level,
                    "propagate": False,
                }
            },
        }
    )