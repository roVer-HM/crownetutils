import socket
import time
from roveranalyzer.utils import logger

SOCKET_TIMEOUT = 10


def check(host: str, port: int, time_to_wait: int = 30, retry_timeout: int = 3) -> bool:
    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    test_socket.settimeout(SOCKET_TIMEOUT)

    start_time = time.time()
    end_time = start_time + time_to_wait
    success = False

    while time.time() < end_time:
        try:
            # try to establish a TCP connection
            test_socket.connect((host, port))
            success = True
            break
        except socket.error as msg:
            logger.debug(
                f"Server is NOT available: Connection to {host}:{port} failed ({msg})."
            )
            if time.time() < end_time:
                logger.debug("Retrying...")
                time.sleep(retry_timeout)

    if success:
        logger.debug(
            f"Server is available: Connection to {host}:{port} can be established."
        )
    else:
        logger.warn(
            f"Server {host}:{port} is NOT available - retry limit reached - giving up."
        )

    return success
