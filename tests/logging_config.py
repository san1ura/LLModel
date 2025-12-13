import logging
import sys


def setup_test_logging():
    """
    Test logging configuration
    """
    # Create main test logger
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.DEBUG)

    # Don't add handler if already exists
    if logger.handlers:
        return logger

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # Add handler
    logger.addHandler(handler)

    return logger


def log_test_start(test_name):
    """Log test start"""
    logger = logging.getLogger('test_logger')
    logger.info(f"TEST START: {test_name}")


def log_test_step(test_name, step, details=None):
    """Log test step"""
    logger = logging.getLogger('test_logger')
    if details:
        logger.debug(f"TEST STEP: {test_name} - {step} - Details: {details}")
    else:
        logger.debug(f"TEST STEP: {test_name} - {step}")


def log_test_success(test_name, message="Test completed successfully"):
    """Log test success"""
    logger = logging.getLogger('test_logger')
    logger.info(f"TEST SUCCESS: {test_name} - {message}")


def log_test_failure(test_name, error_message):
    """Log test failure"""
    logger = logging.getLogger('test_logger')
    logger.error(f"TEST FAILURE: {test_name} - Error: {error_message}")