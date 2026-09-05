"""Offline regression checks for independent internal/public release docs."""

import os
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
        (self.root / "release.json").write_text('{"stable_tag":"v9.8.7"}')
        (self.root / "CHANGELOG.md").write_text("## v9.8.7 (stable; internal promotion)\n")

    def check_provider(self, provider):
        return subprocess.run(
            ["bash", str(self.script)],
            env={**os.environ, "RELEASE_PROVIDER": provider},
            capture_output=True, text=True,
        ).returncode

    def test_internal_publication_does_not_claim_public_image(self):
        (self.root / "README.md").write_text(
            "| Internal image | `git.home.corenode.com/homelab/sglang-glm53-flash-sm120-container:v9.8.7` |\n"
            "The current internal stable image is `v9.8.7`.\n"
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
        self.assertNotEqual(self.check_provider("forgejo"), 0)

    def test_unknown_provider_fails_closed(self):
        self.assertEqual(self.check_provider("unknown"), 2)


if __name__ == "__main__":
    unittest.main()
