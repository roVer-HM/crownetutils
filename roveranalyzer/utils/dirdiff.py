import csv
import glob
import hashlib
import multiprocessing
import os
import pprint
import re


def create_md(arg):
    """
    Returns: key/value where key is the file path and value is the md5 hash
    item: full path of file from which the hash is build
    remove_front: part of path which is not part of the key use later
    """
    item, remove_front = arg
    if os.path.isdir(item):
        return []  # ignore dirs
        # return item[len(remove_front) :], "dir"
    else:
        md5_hash = hashlib.md5()
        with open(item, "rb") as fd:
            md5_hash.update(fd.read())
    return item[len(remove_front) :], md5_hash.hexdigest()


def create_dir_diff(
    path,
    remove_front="",
    exclude=(r".*sca$", r".*vec$", r".*container.*\.out$", r".*command\.out$"),
):
    dir_content = glob.glob(path, recursive=True)
    dir_content_filtered = []
    ex_pattern = re.compile("|".join(exclude))
    for item in dir_content:
        if ex_pattern.match(item):
            continue
        else:
            dir_content_filtered.append((item, remove_front))

    njobs = int(multiprocessing.cpu_count() * 0.60)
    pool = multiprocessing.Pool(processes=njobs)
    ret = pool.map(create_md, dir_content_filtered)
    ret = {e[0]: e[1] for e in ret if len(e) == 2}

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


def compare_diff(left: dict, right: dict):
    ret = True
    diff = []
    diff.append(["key", "expected Hash", "  ", "computed Hash"])
    for k, v in left.items():
        if k not in right:
            diff.append([k, v, "??", "file not found!"])
            ret = False
        else:
            if v != right[k]:
                diff.append([k, v, "!=", right[k]])
                ret = False
    not_expected_keys = set(right.keys()) - set(left.keys())
    if len(not_expected_keys) > 0:
        for k in not_expected_keys:
            ret = False
            diff.append([k, "file not found!", "??", right[k]])

    err_list = []
    if len(diff) > 0:
        c = [0, 0, 0, 0]
        for row in diff:
            for idx, _ in enumerate(c):
                c[idx] = max(c[idx], len(row[idx]))
        for row in diff:
            err_list.append(
                f"{row[0].ljust(c[0])}  {row[1].ljust(c[1])} {row[2].rjust(c[2])} {row[3].ljust(c[3])}"
            )

    return ret, err_list


if __name__ == "__main__":
    p = "/home/vm-sts/repos/crownet/crownet/tests/fingerprint/hash.d/guiding_crowds/final_test_3.csv"
    p2 = "/home/vm-sts/repos/crownet/crownet/tests/fingerprint/hash.d/guiding_crowds/final_test_3.csv.UPDATED"
    dict1 = csv_to_diffdict(p)
    dict2 = csv_to_diffdict(p2)
    pprint.pprint(compare_diff(dict1, dict2))
