# -*- coding:utf-8 -*-
import json
import yaml
import logging
import logging.config
import os

default_config = {
    "version": 1,
    "incremental": False,
    "disable_existing_loggers": True,
    "formatters": {
        "default": {
            "format": "%(name)s %(asctime)s [%(filename)s %(funcName)s()] <%(levelname)s>: %(message)s",
        },
        "brief": {
            "format": "%(name)s %(asctime)s [%(funcName)s() %(lineno)d] <%(levelname)s>: %(message)s",
        }
    },
    "filters": {},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "brief",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
        "file_detail": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "default",
            "level": "DEBUG",
            "filename": "./log/detail_logger.log",
            "encoding": "utf8",
            "maxBytes": 10485760,
            "backupCount": 10,
        },
        "file_info": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "default",
            "level": "INFO",
            "filename": "./log/info_logger.log",
            "encoding": "utf8",
            "maxBytes": 10485760,
            "backupCount": 10,
        }
    },
    "loggers": {
        "cov2": {
            "level": "DEBUG",
            "handlers": ["console", "file_detail", "file_info"],
            "propagate": False,
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file_detail"],
    },
}


def setup_logging(
        default_log_config=None,
        is_yaml=True,
        default_level=logging.INFO,
        env_key='LOG_CFG',
):
    """Setup logging configuration
    Call this only once from the application main() function or __main__ module!
    This will configure the python logging module based on a logging configuration
    in the following order of priority:
       1. Log configuration file found in the environment variable specified in the `env_key` argument.
       2. Log configuration file found in the `default_log_config` argument.
       3. Default log configuration found in the `logconfig.default_config` dict.
       4. If all of the above fails: basicConfig is called with the `default_level` argument.
    Args:
        default_log_config (Optional[str]): Path to log configuration file.
        env_key (Optional[str]): Environment variable that can optionally contain
            a path to a configuration file.
        default_level (int): logging level to set as default. Ignored if a log
            configuration is found elsewhere.
        is_yaml (bool): weather config file is a yaml file
    Returns: None
    """
    dict_config = None
    logconfig_filename = default_log_config
    env_var_value = os.getenv(env_key, None)

    if env_var_value is not None:
        logconfig_filename = env_var_value

    if default_config is not None:
        dict_config = default_config

    if logconfig_filename is not None and os.path.exists(logconfig_filename):
        with open(logconfig_filename, 'rt', encoding="utf8") as f:
            if is_yaml:
                file_config = yaml.load(f, Loader=yaml.FullLoader)
            else:
                file_config = json.load(f)
        if file_config is not None:
            dict_config = file_config

    if dict_config is not None:
        logging.config.dictConfig(dict_config)
    else:
        logging.basicConfig(level=default_level)


if __name__ == '__main__':
    with open("./logconfig.yaml", "w") as f:
        yaml.dump(default_config, f)