import csv
import glob
import hashlib
import multiprocessing
import os


def create_md(arg):
    item, remove_front = arg
    if os.path.isdir(item):
        return item[len(remove_front) :], "dir"
    else:
        md5_hash = hashlib.md5()
        with open(item, "rb") as fd:
            md5_hash.update(fd.read())
    return item[len(remove_front) :], md5_hash.hexdigest()


def create_dir_diff(path, remove_front="", exclude=()):
    dir_content = glob.glob(path, recursive=True)
    dir_content_filtered = []
    for item in dir_content:
        if any([exclude_str in item for exclude_str in exclude]):
            continue
        else:
            dir_content_filtered.append((item, remove_front))

    njobs = int(multiprocessing.cpu_count() * 0.60)
    pool = multiprocessing.Pool(processes=njobs)
    ret = pool.map(create_md, dir_content_filtered)
    ret = {k: v for k, v in ret}

    return ret


def diffdict_to_csv(path, diff_dict: dict, delimiter=";"):
    with open(path, "w", encoding="utf-8") as fd:
        fd.write(f"path{delimiter}md5_digest\n")
        for p, d in diff_dict.items():
            fd.write(f"{p}{delimiter}{d}\n")


def csv_to_diffdict(path, delimiter=";"):
    with open(path, "r") as fd:
        _csv = csv.reader(fd, delimiter=delimiter)
        next(_csv)  # skip header
        ret = {}
        for line in _csv:
            if len(line) == 2:
                ret[line[0]] = line[1]
    return ret


def compare(left: dict, right: dict, print_err=True):
    ret = True
    for k, v in left.items():
        if k not in right:
            if print_err:
                print(f"{k}\t\t{v} (entry not found in right!)")
            ret = False
        else:
            if v != right[k]:
                if print_err:
                    print(f"{k}\t\t{v} != {right[k]}")
                ret = False
    not_expected_keys = set(right.keys()) - set(left.keys())
    if len(not_expected_keys) > 0:
        if print_err:
            err = "\n   ".join(not_expected_keys)
            print(
                f"found {len(not_expected_keys)} not expected keys in right: \n   {err}"
            )
    return ret


if __name__ == "__main__":
    pass
