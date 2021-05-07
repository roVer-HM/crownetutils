import unittest

from roveranalyzer.entrypoint.parser import ArgList


class ArglistTest(unittest.TestCase):
    def setUp(self) -> None:
        self.arglist = ArgList()

    def compare(self, should, _arg=None):
        _arg = self.arglist if _arg is None else _arg
        self.assertEqual(_arg.to_string(), should)

    def test_add(self):
        self.arglist.add("foo", "bar")
        self.arglist.add("--baz", [1, 2, 3])
        self.compare("foo bar --baz 1 2 3")
        self.arglist.add_override("--baz", ["one", "tow"])
        self.compare("foo bar --baz one tow")

    def test_add2(self):
        self.arglist.add("foo", "bar")
        self.arglist.add("--baz")
        self.compare("foo bar --baz")
        self.arglist.add("run_script.py", pos=0)
        self.compare("run_script.py foo bar --baz")

    def test_add_err(self):
        self.arglist.add("foo", "bar")
        self.arglist.add("--baz", [1, 2, 3])
        self.assertRaises(ValueError, self.arglist.add, "--baz", ["one", "tow"])

    def test_append(self):
        self.arglist.append("-l", "foo")
        self.arglist.append("-l", "bar")
        self.compare("-l foo -l bar")

    def test_add_if_missing(self):
        self.arglist.add_if_missing("-f")
        self.arglist.add_if_missing("-f")
        self.arglist.add_if_missing("-f")
        self.compare("-f")

    def test_update_value(self):
        self.arglist.add("--baz", [1, 2, 3])
        self.compare("--baz 1 2 3")
        self.arglist.update_value("--baz", "one-one")
        self.compare("--baz one-one")
        self.arglist.update_value("--foo", 22)
        self.compare("--baz one-one --foo 22")

    def test_update_idx(self):
        self.arglist.add("a", "b")
        self.arglist.add("c", "D")
        self.arglist.add("y", "z")
        self.compare("a b c D y z")
        self.arglist.update_index(1, "foo")
        self.compare("a b foo y z")

    def test_merge1(self):
        self.arglist.add("a", "b")
        self.arglist.add("c", "D")
        self.arglist.add("y", "z")
        self.compare("a b c D y z")
        other = ArgList()
        other.add("a", "b")
        other.add("c", "d")
        other.add("e", "f")
        self.arglist.merge(other, how="add_override")
        self.compare("y z a b c d e f")

    def test_merge2(self):
        self.arglist.add("a", "b")
        self.arglist.add("c", "D")
        self.arglist.add("y", "z")
        self.compare("a b c D y z")
        other = ArgList()
        other.add("a", "b")
        other.add("c", "d")
        other.add("e", "f")
        self.arglist.merge(other, how="add_if_missing")
        self.compare("a b c D y z e f")

    def test_merge3(self):
        self.arglist.add("a", "b")
        self.arglist.add("c", "D")
        self.arglist.add("y", "z")
        self.compare("a b c D y z")
        other = ArgList()
        other.add("a", "b")
        other.add("c", "d")
        other.add("e", "f")
        self.assertRaises(ValueError, self.arglist.merge, other, "add")

    def test_merge4(self):
        self.arglist.add("a", "b")
        self.arglist.add("c", "D")
        self.arglist.add("y", "z")
        self.compare("a b c D y z")
        other = ArgList()
        other.add("a", "b")
        other.add("c", "d")
        other.add("e", "f")
        self.assertRaises(
            RuntimeError, self.arglist.merge, other, "some-wrong-function"
        )

    def test_from_list(self):
        data = [["-f", "blaa"], ["-vv", None], ["-j", 80]]
        arg_list = ArgList.from_list(data)
        self.compare("-f blaa -vv -j 80", arg_list)

    def test_get_value(self):
        data = [["-f", "blaa"], ["-vv", None], ["-j", 80]]
        arg_list = ArgList.from_list(data)
        self.assertEqual(arg_list.get_value("-f"), "blaa")
        self.assertEqual(arg_list.get_value("-f", "default"), "blaa")
        self.assertEqual(arg_list.get_value("-XXX", "foo"), "foo")
        self.assertEqual(arg_list.get_value("-XXX"), None)

    def test_from_flat_list(self):
        flat_list = "--foo bar -i 12 -i 44 -vvv --baz aa cc dd".split()
        arg_list = ArgList.from_flat_list(flat_list)
        self.compare("--foo bar -i 12 -i 44 -vvv --baz aa cc dd", arg_list)
        self.assertEqual(
            [
                ["--foo", "bar"],
                ["-i", "12"],
                ["-i", "44"],
                ["-vvv", None],
                ["--baz", ["aa", "cc", "dd"]],
            ],
            arg_list.raw(),
        )

    def test_from_flat_list2(self):
        flat_list = "/bin/bash ./fancy-script.sh --foo bar -i 12 -i 44 -vvv --baz aa cc dd".split()
        arg_list = ArgList.from_flat_list(flat_list)
        self.compare(
            "/bin/bash ./fancy-script.sh --foo bar -i 12 -i 44 -vvv --baz aa cc dd",
            arg_list,
        )
        self.assertEqual(
            [
                ["/bin/bash", None],
                ["./fancy-script.sh", None],
                ["--foo", "bar"],
                ["-i", "12"],
                ["-i", "44"],
                ["-vvv", None],
                ["--baz", ["aa", "cc", "dd"]],
            ],
            arg_list.raw(),
        )
