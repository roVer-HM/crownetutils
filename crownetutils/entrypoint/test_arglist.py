import unittest

from crownetutils.entrypoint.parser import ArgList


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

    def test_remove_startswith(self):
        arg_list = ArgList.from_string("-f omnetpp.ini -c foobar --foo=bar")
        self.assertTrue(arg_list.contains_key("--foo=bar"))
        self.assertTrue(arg_list.contains_key("-c"))
        arg_list.remove_key_startswith("--foo")
        self.assertFalse(arg_list.contains_key("--foo=bar"))
        self.assertTrue(arg_list.contains_key("-c"))

    def test_contains_startswith(self):
        arg_list = ArgList.from_string("-f omnetpp.ini -c foobar --foo=bar")
        self.assertTrue(arg_list.contains_key_startswith("--foo"))

    def test_getvalue_equal(self):
        arg_list = ArgList.from_string("-f omnetpp.ini -c foobar --foo=bar")
        self.assertEqual(arg_list.get_value("--foo="), "bar")

    def test_has_command(self):
        arg_list_1 = ArgList.from_string("rm -rf /*")
        self.assertTrue(arg_list_1.has_command)

        arg_list_2 = ArgList.from_string("-f omnetpp.ini -c foobar")
        self.assertFalse(arg_list_2.has_command)

        arg_list_3 = ArgList.from_string("--f omnetpp.ini -c foobar")
        self.assertFalse(arg_list_3.has_command)

    def test_from_string(self):
        cmd = (
            "../../src/run_crownet_dbg -u Cmdenv -f omnetpp.ini -c fTestDcD -r 0 --sim-time-limit=80s "
            '"--fingerprint=65f7-c707/tplx 65f7-c88/tplx" --cpu-time-limit=900s --vector-recording=false '
            '--scalar-recording=true  --result-dir=/crownet.csv/_fooBarD_r_0 --foo "a b c"'
        )
        arg_list = ArgList.from_string(cmd)
        self.assertEqual(
            [
                ["../../src/run_crownet_dbg", None],
                ["-u", "Cmdenv"],
                ["-f", "omnetpp.ini"],
                ["-c", "fTestDcD"],
                ["-r", "0"],
                ["--sim-time-limit=80s", None],
                [
                    "--fingerprint=65f7-c707/tplx 65f7-c88/tplx",
                    None,
                ],  # space in key is allowed
                ["--cpu-time-limit=900s", None],
                ["--vector-recording=false", None],
                ["--scalar-recording=true", None],
                ["--result-dir=/crownet.csv/_fooBarD_r_0", None],
                ["--foo", "a b c"],  # space in value is allowed
            ],
            arg_list.raw(),
        )
