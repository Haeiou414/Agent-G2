import unittest

from reproduction.verify_gpu_environment import parse_nvidia_smi


class GpuPreflightTests(unittest.TestCase):
    def test_parses_multiple_gpu_rows(self) -> None:
        rows = parse_nvidia_smi(
            "NVIDIA RTX 4090, 24564, 570.00\nNVIDIA A100-SXM4-80GB, 81920, 570.00\n"
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["name"], "NVIDIA RTX 4090")
        self.assertEqual(rows[0]["memory_mib"], 24564)
        self.assertEqual(rows[1]["memory_mib"], 81920)

    def test_rejects_unexpected_nvidia_smi_output(self) -> None:
        with self.assertRaises(ValueError):
            parse_nvidia_smi("GPU without requested CSV fields")


if __name__ == "__main__":
    unittest.main()
