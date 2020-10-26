import os
import random
import shutil
import tarfile
import urllib.request
import zipfile
from string import ascii_lowercase


class Downloader:
    def __init__(self, url):
        self.url = url

    def download(self, path, override=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if override:
            print(f"load file: {path}")
            urllib.request.urlretrieve(self.url, path)
        else:
            if os.path.exists(path):
                print(f"existing archive found. {path}")
            else:
                print(f"load file: {path}")
                urllib.request.urlretrieve(self.url, path)

    def extract(self, archive_dir, dir, **kwargs):
        raise ValueError("Implement")


class TarDownloader(Downloader):
    def __init__(self, url, mode):
        super().__init__(url)
        self.mode = mode

    def extract(self, archive_dir, data_dir, **kwargs):
        with open(archive_dir, "rb") as archive:
            tar = tarfile.open(fileobj=archive, mode=self.mode)
            print(f"extract to : {data_dir}")
            tar.extractall(path=data_dir)


class ZipDownloader(Downloader):
    def __init__(self, url):
        super().__init__(url)

    def extract(self, archive_dir, data_dir, **kwargs):
        with zipfile.ZipFile(archive_dir, "r") as archive:
            print(f"extract to : {data_dir}")
            archive.extractall(data_dir)


class LocalDownloader(Downloader):
    def __init__(self):
        super().__init__("")

    def download(self, path, override=False):
        pass

    def extract(self, archive_dir, dir, **kwargs):
        pass


class TestDataHandler:
    @classmethod
    def local(cls, path):
        d = LocalDownloader()
        base, fname = os.path.split(path)
        return cls(
            downloader=d,
            file_name=fname,
            suffix="",
            archive_base_dir="",
            extract_to=base,
            keep_archive=True,
        )

    @classmethod
    def tar(
        cls,
        url,
        file_name="",
        suffix="tar.gz",
        mode="r:gz",
        archive_base_dir="",
        extract_to="/tmp/roveranalyzer",
        keep_archive=False,
    ):
        d = TarDownloader(url, mode)
        return cls(
            downloader=d,
            file_name=file_name,
            suffix=suffix,
            archive_base_dir=archive_base_dir,
            extract_to=extract_to,
            keep_archive=keep_archive,
        )

    @classmethod
    def zip(
        cls,
        url,
        file_name="",
        suffix="zip",
        archive_base_dir="",
        extract_to="/tmp/roveranalyzer",
        keep_archive=False,
    ):
        d = ZipDownloader(url)
        return cls(
            downloader=d,
            file_name=file_name,
            suffix=suffix,
            archive_base_dir=archive_base_dir,
            extract_to=extract_to,
            keep_archive=keep_archive,
        )

    def __init__(
        self,
        downloader,
        file_name="",
        suffix="tar.gz",
        archive_base_dir="",
        extract_to="/tmp/roveranalyzer",
        keep_archive=False,
    ):
        """
        :url:               download archived test data
        :filename:          name for the downloaded archive.
        :suffix:            expected suffix of archive file
        :mode:              tarfile mode string to extract default: "r:gz" for tar.gz files
        :archive_base_dir:  Top level directory *within* the archive. Empty if all data is on the top level
        :extract_to:        Base path where the data should be extracted to. Defaults to /tmp
        :keep_archive:      Delete archive after extraction.
        """
        self.downloader: Downloader = downloader
        self.suffix = suffix
        self.archive_base_dir = archive_base_dir
        self.extract_base_path = extract_to
        self.keep_archive = keep_archive
        if len(file_name) == 0:
            self.file_name = "".join(random.choice(ascii_lowercase) for i in range(6))
            self.data_dir = os.path.join(self.extract_base_path, f"{self.file_name}.d")
        else:
            self.file_name = file_name
            self.data_dir = os.path.join(self.extract_base_path, self.file_name)

        # build paths
        self.archive_path = os.path.join(
            self.extract_base_path, f"{self.file_name}.{self.suffix}"
        )

    def download_test_data(self, override=True):
        # get archive file
        self.downloader.download(self.archive_path, override=override)

        # extract files
        self.downloader.extract(self.archive_path, self.data_dir)

        # cleanup
        if not self.keep_archive:
            print(f"removing archive {self.archive_path}")
            os.remove(self.archive_path)

    def remove_data(self):
        print(f"remove data in {self.data_dir}")
        shutil.rmtree(self.data_dir)

    def abs_path(self, path):
        return os.path.abspath(self.join(path))

    def join(self, path):
        if type(path) == str:
            path = [path]
        return os.path.join(self.data_dir, self.archive_base_dir, *path)
