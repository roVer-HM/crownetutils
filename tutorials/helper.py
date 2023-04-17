import glob
import os

from tests.utils import TestDataHandler


def check_data():
    data = TestDataHandler.zip(
        url="https://sam.cs.hm.edu/samcloud/index.php/s/ra6KTqinCX5WpWy/download",
        file_name="test_data",
        extract_to=".",
    )
    data.download_test_data(override=False)

    files = glob.glob("./test_data.d/**", recursive=True)
    print(f"{os.path.abspath('.')}:")
    for f in files:
        print(f"  |-- {f[2:]}")

    return os.path.abspath(data.data_dir)
