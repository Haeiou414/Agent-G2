import ast
import unittest
from pathlib import Path

from reproduction.audit_config import REPO_ROOT


PREPARE = REPO_ROOT / "examples" / "data_preprocess" / "prepare.py"


class TextPlaceholderDataTests(unittest.TestCase):
    def test_text_mode_builds_local_placeholder_rows(self) -> None:
        source = PREPARE.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("datasets.Dataset.from_list", source)
        self.assertIn("range(args.train_data_size)", source)
        self.assertIn("range(args.val_data_size)", source)
        self.assertIn("if args.mode == 'text':", source)

    def test_geometry_download_is_confined_to_visual_branch(self) -> None:
        source = PREPARE.read_text(encoding="utf-8")
        text_branch = source.index("if args.mode == 'text':")
        visual_branch = source.index("else:", text_branch)
        download = source.index("datasets.load_dataset('hiyouga/geometry3k')")
        self.assertLess(text_branch, visual_branch)
        self.assertLess(visual_branch, download)


if __name__ == "__main__":
    unittest.main()
