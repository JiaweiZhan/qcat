import sys
from loguru import logger

def setLogger(
    level='INFO',
    format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    stream=sys.stderr,  # Default to logging to stderr
    logfile=None,
    rotation='10 MB',
    retention='30 days',
    compression='zip',
    filter_out=None,
    ):
    logger.remove()  # Remove default handler

    def filter_out_module(record):
        # Exclude all logs from 'qcat.assignGrid.assignGrid'
        if filter_out and filter_out in record["name"]:
            return False
        else:
            return True

    if logfile:
        logger.add(
            sink=logfile,
            level=level,
            format=format,
            rotation=rotation,
            retention=retention,
            compression=compression,
            filter_out=filter_out_module,
        )

    # Add a stream handler
    logger.add(
        stream,
        level=level,
        format=format,
        filter=filter_out_module,
    )
