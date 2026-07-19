from loguru import logger
import sys

# Configure structured logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("logs/agent_audit.log", rotation="500 MB")

def log_trace(step: str, metadata: dict):
    logger.info(f"Step: {step} | Data: {metadata}")