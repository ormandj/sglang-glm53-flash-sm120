"""Offline regression checks for independent internal/public release docs."""

import os
import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class PublicationDocsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        self.script = self.root / "scripts/validate-publish-release.sh"
        shutil.copyfile(Path(__file__).with_name(self.script.name), self.script)
        shutil.copyfile(Path(__file__).with_name("render-release-changes.py"), self.root / "scripts/render-release-changes.py")
        (self.root / "release.json").write_text('{"stable_tag":"v9.8.7"}')
        (self.root / "CHANGELOG.md").write_text("## v9.8.7 (stable; internal promotion)\n\n- Fix the documented serving failure.\n")

    def check_provider(self, provider):
        return subprocess.run(
            ["bash", str(self.script)],
            env={**os.environ, "RELEASE_PROVIDER": provider},
            capture_output=True, text=True,
        ).returncode

    def test_internal_publication_does_not_claim_public_image(self):
        (self.root / "README.md").write_text(
            "| Image | `ghcr.io/ormandj/sglang-glm53-flash-sm120:v9.8.6` |\n"
            "The current published stable image is `v9.8.6`.\n"
        )
        self.assertEqual(self.check_provider("forgejo"), 0)
        self.assertNotEqual(self.check_provider("github"), 0)

    def test_public_publication_requires_current_public_tag(self):
        (self.root / "README.md").write_text(
            "| Image | `ghcr.io/ormandj/sglang-glm53-flash-sm120:v9.8.7` |\n"
            "The current published stable image is `v9.8.7`.\n"
        )
        self.assertEqual(self.check_provider("github"), 0)
        self.assertEqual(self.check_provider("forgejo"), 0)

    def test_unknown_provider_fails_closed(self):
        self.assertEqual(self.check_provider("unknown"), 2)


class ReleaseChangesTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("release_changes", Path(__file__).with_name("render-release-changes.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.render = module.release_changes

    def test_exact_version_and_subsections(self):
        text = "## v1.2.30 (stable; other)\n\n- Wrong version.\n\n## v1.2.3 (stable; release)\n\n- Fix serving.\n\n### Known issues\n\n- Keep this warning.\n\n## v1.2.3-rc.1\n\n- Candidate history.\n"
        self.assertEqual(self.render(text, "v1.2.3"), "- Fix serving.\n\n### Known issues\n\n- Keep this warning.\n")

    def test_long_prose_is_not_hard_wrapped(self):
        bullet = "- " + "release change " * 20
        self.assertEqual(self.render("## v1.2.3\n\n" + bullet, "v1.2.3"), bullet.rstrip() + "\n")

    def test_internal_housekeeping_is_rejected(self):
        for note in (
            "Promote without rebuilding within each registry.",
            "The qualified internal digest is sha256:123.",
            "Image: git.home.corenode.com/homelab/image:v1.2.3.",
            "The ordered-marker oracle passed.",
            "Receipts live in the primary project repository.",
        ):
            with self.subTest(note=note), self.assertRaises(ValueError):
                self.render("## v1.2.3\n\n- " + note, "v1.2.3")

    def test_missing_empty_duplicate_and_prerelease_sections_fail(self):
        for text, tag in (
            ("## v1.2.30\n- Other.\n", "v1.2.3"),
            ("## v1.2.3\n", "v1.2.3"),
            ("## v1.2.3\nRead CHANGELOG.md.\n", "v1.2.3"),
            ("## v1.2.3\n- One.\n## v1.2.3\n- Two.\n", "v1.2.3"),
            ("## v1.2.3-rc.1\n- Candidate.\n", "v1.2.3-rc.1"),
        ):
            with self.subTest(text=text, tag=tag), self.assertRaises(ValueError):
                self.render(text, tag)


if __name__ == "__main__":
    unittest.main()
