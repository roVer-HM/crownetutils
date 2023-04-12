import subprocess
import time

from roveranalyzer.utils.logging import logger

SOCKET_TIMEOUT = 10


def check(
    container: str, port: int, time_to_wait: int = 30, retry_timeout: int = 3
) -> bool:
    """Wait for Docker container startup by polling expected port.

    Args:
        container (str): Docker container full name
        port (int): Expected port exposed by Docker container.
        time_to_wait (int, optional): Timeout in seconds. Defaults to 30 s.
        retry_timeout (int, optional): Sleep time before polling again in seconds. Defaults to 3 s.

    Returns:
        bool: True if container is reachable and False if timeout reached.
    """

    start_time = time.time()
    end_time = start_time + time_to_wait
    success = False

    while time.time() < end_time and not success:
        try:
            # check if a process is listening on the specified port
            result = subprocess.check_output(
                [
                    "docker",
                    "exec",
                    container,
                    "bash",
                    "-c",
                    f'cat /proc/net/tcp | grep ":{port:04X} " >/dev/null ; echo $?',
                ]
            )
            result_str = result.decode("utf-8")
            if result_str == "0\n":
                success = True
            else:
                logger.debug(
                    f"Server is NOT available: on container {container} no process is listening on port {port} ({result_str})."
                )
        except subprocess.CalledProcessError as e:
            logger.debug(
                f"Server is NOT available: Subprocess error for container {container}:{port}: ({e})."
            )
        if time.time() < end_time and not success:
            logger.debug("Retrying...")
            time.sleep(retry_timeout)

    if success:
        logger.debug(
            f"Server on container {container} is available: Connection to port {port} can be established."
        )
    else:
        logger.warn(
            f"Server on container {container}:{port} is NOT available - retry limit reached - giving up."
        )

    return success
