import time
import subprocess
from roveranalyzer.utils import logger

SOCKET_TIMEOUT = 10


def check(
    container: str, port: int, time_to_wait: int = 30, retry_timeout: int = 3
) -> bool:

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
