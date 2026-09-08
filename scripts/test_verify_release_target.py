"""Exercise release-target guards with real Git histories."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name('verify-release-target.sh').resolve()

class ReleaseTargetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.env = dict(os.environ, GIT_AUTHOR_NAME='Test', GIT_AUTHOR_EMAIL='test@example.invalid',
                        GIT_COMMITTER_NAME='Test', GIT_COMMITTER_EMAIL='test@example.invalid')
        self.git('init', '-q')
        self.target = self.commit('README.md', 'Reviewed release documentation\n')

    def git(self, *args):
        return subprocess.check_output(['git', '-c', 'commit.gpgsign=false', *args], cwd=self.root, env=self.env, text=True).strip()

    def commit(self, path, value):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value)
        self.git('add', path)
        self.git('commit', '-qm', 'Fixture')
        return self.git('rev-parse', 'HEAD')

    def check(self, target=None, workflow=None):
        return subprocess.run(['bash', str(SCRIPT), target or self.target,
                               workflow or self.git('rev-parse', 'HEAD')],
                              cwd=self.root, env=self.env, capture_output=True).returncode

    def test_current_commit_allowed(self):
        self.assertEqual(self.check(), 0)

    def test_publication_only_descendant_allowed(self):
        self.commit('.github/workflows/publish-release.yml', 'Repaired workflow\n')
        self.commit('scripts/verify-release-target.sh', 'Repaired guard\n')
        self.assertEqual(self.check(), 0)

    def test_changed_release_documents_rejected(self):
        self.commit('README.md', 'Different release documentation\n')
        self.assertNotEqual(self.check(), 0)

    def test_changed_image_inputs_rejected(self):
        self.commit('Containerfile', 'FROM different\n')
        self.assertNotEqual(self.check(), 0)

    def test_unrelated_workflow_rejected(self):
        self.commit('.github/workflows/build-image.yml', 'Changed build\n')
        self.assertNotEqual(self.check(), 0)

    def test_divergent_release_rejected(self):
        other = self.commit('README.md', 'Other branch\n')
        self.git('checkout', '--detach', '-q', self.target)
        self.commit('.github/workflows/publish-release.yml', 'Workflow\n')
        self.assertNotEqual(self.check(target=other), 0)

    def test_abbreviated_or_missing_release_rejected(self):
        self.assertNotEqual(self.check(target=self.target[:12]), 0)
        self.assertNotEqual(self.check(target='0' * 40), 0)

    def test_wrong_workflow_checkout_rejected(self):
        self.commit('.github/workflows/publish-release.yml', 'Workflow\n')
        self.assertNotEqual(self.check(workflow=self.target), 0)

if __name__ == '__main__':
    unittest.main()
