import io
import subprocess
import sys
import unittest
from unittest.mock import patch

from fw_jsbgym.visualizers.visualizer import PlotVisualizer, _close_process, _wait_for_visualizer


class VisualizerProcessTests(unittest.TestCase):
    def test_failed_start_reports_error_instead_of_waiting_forever(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "print('startup failed')"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        with self.assertRaisesRegex(RuntimeError, "exited before it was ready"):
            _wait_for_visualizer(process, "ready", "Test visualizer")
        self.assertIsNotNone(process.poll())
        self.assertTrue(process.stdout.closed)

    def test_close_stops_and_reaps_visualizer_process(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        _close_process(process)
        self.assertIsNotNone(process.poll())
        self.assertTrue(process.stdout.closed)

    def test_animated_plot_uses_current_python_and_handles_spaces(self):
        process = unittest.mock.MagicMock()
        process.stdout = io.BytesIO(b"Animation plot started...\n")
        process.poll.return_value = 0
        with patch(
            "fw_jsbgym.visualizers.visualizer.resource_filename", return_value="/tmp/plot script.py",
        ), patch("fw_jsbgym.visualizers.visualizer.subprocess.Popen", return_value=process) as popen:
            visualizer = PlotVisualizer(True, "ACExample-v0", "/tmp/telemetry file.csv")
            visualizer.close()
        self.assertEqual(
            popen.call_args.args[0],
            [sys.executable, "/tmp/plot script.py", "--tele-file", "/tmp/telemetry file.csv", "--animate"],
        )


if __name__ == "__main__":
    unittest.main()
