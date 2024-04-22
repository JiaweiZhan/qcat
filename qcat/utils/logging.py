import sys
from loguru import logger

def setLogger(
    level='INFO',
    format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    stream=sys.stderr,  # Default to logging to stderr
    logfile=None,
    rotation='10 MB',
    retention='30 days',
    compression='zip'
    ):
    logger.remove()  # Remove default handler

    if logfile:
        logger.add(
            sink=logfile,
            level=level,
            format=format,
            rotation=rotation,
            retention=retention,
            compression=compression
        )

    # Add a stream handler
    logger.add(
        stream,
        level=level,
        format=format
    )
