import logging
import os
from datetime import datetime

import sys

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_path = os.path.join(os.getcwd(), "logs")

os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


# [ 2024-12-18 21:56:09,233 ] 15         root     - ERROR         - Error occured in python script name [d:\a27_YEARS_OLD\MLOPS_AWS_SAGEMAKER\test.py] line number [11] error message [division by zero]
# [ %(asctime)s             ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
