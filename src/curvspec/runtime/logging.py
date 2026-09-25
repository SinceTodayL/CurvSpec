import logging
from pathlib import Path


def create_logger(output_directory, filename="log.txt"):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("curvspec")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_directory / filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
