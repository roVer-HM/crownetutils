import os
import shutil
import unittest

from roveranalyzer.vadereanalyzer.vadere_project import VadereProject


class VadereProjectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_path = os.path.dirname(__file__)

    def abs_path(self, *p):
        return os.path.abspath(os.path.join(self.base_path, *p))

    def test_project_name(self):
        project = VadereProject(self.abs_path("testData/s2ucre"))
        self.assertEqual(project.project_name, "s2ucre_scenarios")

    def test_wrong_project_dir(self):
        self.assertRaises(
            FileNotFoundError, VadereProject, self.abs_path("testData/s2uSSScre")
        )

    def test_no_output_dir(self):
        shutil.rmtree(
            self.abs_path("testData/s2ucreInvalid", "output"), ignore_errors=True
        )
        self.assertFalse(
            os.path.exists(self.abs_path("testData/s2ucreInvalid", "output"))
        )
        p = VadereProject(self.abs_path("testData/s2ucreInvalid"))
        self.assertTrue(
            os.path.exists(self.abs_path("testData/s2ucreInvalid", "output"))
        )

    def test_load_output_dir(self):
        project = VadereProject(self.abs_path("testData/s2ucre"))
        self.assertEqual(len(project.err_info()), 0)
        self.assertEqual(len(project.output_dirs), 19)

    def test_scenaio_files(self):
        project = VadereProject(self.abs_path("testData/s2ucre"))
        self.assertEqual(
            [
                self.abs_path(
                    "testData/s2ucre/scenarios/bridge_coordinates_kai.scenario"
                ),
                self.abs_path(
                    "testData/s2ucre/scenarios/bridge_coordinates_kai_navigation.scenario"
                ),
                self.abs_path(
                    "testData/s2ucre/scenarios/bridge_coordinates_kai_navigation_random_pos.scenario"
                ),
                self.abs_path(
                    "testData/s2ucre/scenarios/bridge_coordinates_kai_origin_0.scenario"
                ),
                self.abs_path(
                    "testData/s2ucre/scenarios/bridge_coordinates_kai_origin_0_navigation.scenario"
                ),
                self.abs_path(
                    "testData/s2ucre/scenarios/bridge_coordinates_kai_origin_0_navigation_random_pos.scenario"
                ),
                self.abs_path(
                    "testData/s2ucre/scenarios/bridge_coordinates_kai_origin_0_random_pos.scenario"
                ),
                self.abs_path(
                    "testData/s2ucre/scenarios/bridge_coordinates_kai_random_pos.scenario"
                ),
                self.abs_path("testData/s2ucre/scenarios/empty.scenario"),
            ],
            project.scenario_files,
        )


if __name__ == "__main__":
    unittest.main()
