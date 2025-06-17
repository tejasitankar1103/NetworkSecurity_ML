import logging
import os
from datetime import datetime

#log file creation
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
#log path 
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path,exist_ok=True) #directory creation for log

## Total log path
LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)