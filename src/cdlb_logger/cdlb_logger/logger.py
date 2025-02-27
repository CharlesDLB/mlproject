import logging
import logging.handlers
from pythonjsonlogger import jsonlogger

def setup_logger(name, level=logging.INFO, stream:bool=True, file:bool=False, log_file_path:str="logs/global_logger.log") -> logging.Logger:

    # Formatter
    streamformatter = logging.Formatter("%(asctime)s | %(levelname)s [%(name)s|L%(lineno)d] : %(message)s",
                                        datefmt='%Y-%m-%d %H:%M:%S')
    
    jsonformatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(module)s %(lineno)d %(message)s",
                                             datefmt='%Y-%m-%d %H:%M:%S')

    # Logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Stream Handler
    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(streamformatter)
        if not logger.hasHandlers():
            logger.addHandler(stream_handler)

    # File Handler
    if file:
        file_handler = logging.handlers.RotatingFileHandler(filename=log_file_path,
                                                        maxBytes=1000000,
                                                        backupCount=3
                                                        )
        file_handler.setFormatter(jsonformatter)
        if not logger.hasHandlers():
            logger.addHandler(file_handler)

    return logger