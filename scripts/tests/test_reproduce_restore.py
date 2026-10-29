"""Restore safety checks: existing data, path boundaries, aliases and delta priority."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("reproduce", Path(__file__).resolve().parents[1] / "reproduce.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RestoreSafety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dest = self.root / "clone"
        self.mapping = [{"member_prefix": "Project/data/", "restore_prefix": "data/"}]

    def tearDown(self):
        self.temp.cleanup()

    def archive(self, name, rows):
        path = self.root / name
        with tarfile.open(path, "w") as tar:
            for member, value, link in rows:
                info = tarfile.TarInfo(member)
                if link is not None:
                    info.type = tarfile.SYMTYPE
                    info.linkname = link
                    tar.addfile(info)
                else:
                    info.size = len(value)
                    tar.addfile(info, io.BytesIO(value))
        Path(str(path) + ".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
        return path

    def test_existing_data_is_never_overwritten(self):
        archive = self.archive("base.tar", [("Project/data/value", b"saved", None)])
        module.restore(archive, None, self.dest, self.mapping)
        module.restore(archive, None, self.dest, self.mapping)
        target = self.dest / "data/value"
        target.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            module.restore(archive, None, self.dest, self.mapping)
        self.assertEqual(target.read_bytes(), b"existing")

    def test_path_traversal_is_rejected(self):
        archive = self.archive("unsafe.tar", [("Project/data/../../outside", b"bad", None)])
        with self.assertRaises(ValueError):
            module.restore(archive, None, self.dest, self.mapping)
        self.assertFalse((self.root / "outside").exists())

    def test_existing_symlink_cannot_escape_clone(self):
        self.dest.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (self.dest / "data").symlink_to(outside, target_is_directory=True)
        archive = self.archive("base.tar", [("Project/data/value", b"bad", None)])
        with self.assertRaises(ValueError):
            module.restore(archive, None, self.dest, self.mapping)
        self.assertFalse((outside / "value").exists())

    def test_reviewed_alias_replaces_archive_absolute_link(self):
        archive = self.archive("base.tar", [("Project/data/real/value", b"saved", None),
                                             ("Project/data/alias", b"", "/old/private/path")])
        links = [{"path": "data/alias", "target": "data/real"}]
        module.restore(archive, None, self.dest, self.mapping, links=links)
        self.assertEqual((self.dest / "data/alias/value").read_bytes(), b"saved")
        self.assertFalse((self.dest / "data/alias").readlink().is_absolute())
        module.restore_links(self.dest, links)
        with self.assertRaises(ValueError):
            module.restore_links(self.dest, [{"path": "data/bad", "target": "../escape"}])

    def test_unknown_archive_link_is_rejected(self):
        archive = self.archive("bad.tar", [("Project/data/link", b"", "/private/path")])
        with self.assertRaises(ValueError):
            module.restore(archive, None, self.dest, self.mapping)

    def test_delta_wins_and_transient_markers_are_skipped(self):
        base = self.archive("base.tar", [("Project/data/value", b"old", None),
                                         ("Project/data/running.json", b"stale", None)])
        delta = self.archive("delta.tar", [("Project/data/value", b"new", None)])
        module.restore(base, None, self.dest, self.mapping, overlays=[delta])
        self.assertEqual((self.dest / "data/value").read_bytes(), b"new")
        self.assertFalse((self.dest / "data/running.json").exists())

    def test_corrupt_archive_is_rejected_before_extraction(self):
        archive = self.archive("base.tar", [("Project/data/value", b"saved", None)])
        with archive.open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaises(ValueError):
            module.restore(archive, None, self.dest, self.mapping)
        self.assertFalse(self.dest.exists())


if __name__ == "__main__":
    unittest.main()
