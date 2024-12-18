from src.logger.logging import logging
from src.exception.exception import customException
import sys

# logging.info("Welcome to our custom log")


try:
    a = 1
    b = 0
    c = a/b
except Exception as e:
    # raise logging.info(customException(e, sys))
    error = customException(e, sys)
    logging.info(error)
