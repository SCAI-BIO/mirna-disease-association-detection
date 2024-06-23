import logging
from logging import Logger
import optuna

import transformers

def get_custom_logger(name: str, level: int=logging.DEBUG) -> Logger: 
  """
  Get project specific custom logger.

  Example usage:
    import logging
    logger = get_custom_logger(__name__, level=logging.INFO)

  Args:
      name (str): Name of the logger
      level ([int], optional): logging level. Defaults to logging.DEBUG.

  Returns:
      [Logger]: Project logger
  """
  # create formatter
  # formatter = logging.Formatter(fmt='%(asctime)s %(filename)s %(module)s: %(levelname)8s %(message)s')
  # date_format = '%m-%d %H:%M:%S'
  # formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s', datefmt = date_format)
  formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d :: %(funcName)20s()} -  %(levelname)s - %(message)s')
  # colorlog.ColoredFormatter(
  #   "%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s"
  # )

  # create console handler and set level to debug
  handler = logging.StreamHandler()
  handler.setFormatter(formatter)
  
  # create logger
  logger = logging.getLogger(name)
  logger.setLevel(level)
  logger.propagate = False
  #logger.handlers = []
  logger.addHandler(handler)

  return logger

def reset_transformers_logger(logger: Logger):
  handlers = transformers.utils.logging._get_library_root_logger().handlers
  for handler in handlers:
    # formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
    formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d :: %(funcName)20s()} -  %(levelname)s - %(message)s')
    app_formatter = logger.handlers[0].formatter
    handler.setFormatter(app_formatter) if isinstance(app_formatter, logging.Formatter) else handler.setFormatter(formatter)

def reset_optuna_logger(logger: Logger):
  handlers = optuna.logging._get_library_root_logger().handlers
  for handler in handlers:
    # formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
    formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d :: %(funcName)20s()} -  %(levelname)s - %(message)s')
    app_formatter = logger.handlers[0].formatter
    handler.setFormatter(app_formatter) if isinstance(app_formatter, logging.Formatter) else handler.setFormatter(formatter)
