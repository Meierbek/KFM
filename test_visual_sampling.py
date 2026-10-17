"""Synthetic regression tests for aspect-preserving local vision inputs."""
from pathlib import Path
from contextlib import contextmanager
import shutil
import uuid
import unittest

from PIL import Image

from visual_sampling import storyboard, sample_windows


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(__file__).resolve().parent / ".qa"
        self.scratch.mkdir(exist_ok=True)

    @contextmanager
    def temporary(self):
        folder = self.scratch / ("sampling-" + uuid.uuid4().hex)
        folder.mkdir()
        try:
            yield folder
        finally:
            if folder.resolve().parent == self.scratch.resolve():
                shutil.rmtree(folder)

    def test_portrait_uses_cell_width_without_large_black_columns(self):
        with self.temporary() as temp:
            root = Path(temp)
            source = root / "portrait.jpg"
            Image.new("RGB", (360, 640), "red").save(source)
            target = storyboard([source] * 9, list(range(9)), root / "board.jpg")
            with Image.open(target) as board:
                self.assertEqual(board.size, (960, 1797))
                for x in (10, 160, 310, 330, 630, 650, 950):
                    self.assertGreater(board.getpixel((x, 200))[0], 200)

    def test_hero_keeps_temporal_neighbors_and_original_indices(self):
        with self.temporary() as temp:
            root = Path(temp)
            paths = []
            for index in range(7):
                path = root / f"{index}.png"
                Image.new("RGB", (360, 640), (20 + index * 30, 40, 60)).save(path)
                paths.append(path)
            target = storyboard(paths, list(range(7)), root / "hero.jpg", hero=3)
            with Image.open(target) as board:
                self.assertEqual(board.size, (1232, 1455))
                self.assertAlmostEqual(board.getpixel((300, 300))[0], 110, delta=5)
                # Frame 3 has its own large view; frame 4 follows frame 2.
                self.assertAlmostEqual(board.getpixel((800, 100))[0], 20, delta=5)
                self.assertAlmostEqual(board.getpixel((1100, 600))[0], 140, delta=5)

    def test_windows_cover_start_end_and_do_not_exceed_duration(self):
        for duration in (.01, 2, 24, 25, 142.5, 300):
            windows = sample_windows(duration)
            self.assertEqual(windows[0][0], 0)
            self.assertLess(windows[-1][-1], duration)
            self.assertLessEqual(duration-windows[-1][-1], .101)
            for left, right in zip(windows, windows[1:]):
                self.assertLessEqual(right[0], left[-1])


if __name__ == "__main__":
    unittest.main()
