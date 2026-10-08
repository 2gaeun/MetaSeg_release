"""Input contracts and the public run.sh entrypoint, without model inference."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

from bms_deploy.dicom_input import read_input_csv


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv = self.root / 'input.csv'

    def read(self, text, mode='nifti'):
        self.csv.write_text(text)
        return read_input_csv(self.root, self.csv, mode)

    def test_nifti_file_ids_both_extensions_and_nested_directory(self):
        (self.root/'A.nii').touch()
        (self.root/'nested').mkdir()
        (self.root/'nested/B.nii.gz').touch()
        rows = self.read('\nfile_id,sequence\nA,t1ce\nB,bb\n')
        self.assertEqual(rows, [('A', 'T1CE', self.root/'A.nii'),
                                ('B', 'BB', self.root/'nested/B.nii.gz')])
        self.assertEqual(self.read('A,BB\n')[0][0], 'A')

    def test_missing_or_ambiguous_nifti_ids(self):
        with self.assertRaises(FileNotFoundError):
            self.read('missing,BB\n')
        (self.root/'A.nii').touch()
        (self.root/'A.nii.gz').touch()
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            self.read('A,BB\n')
        (self.root/'A.nii.gz').unlink()
        (self.root/'nested').mkdir()
        (self.root/'nested/A.nii').touch()
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            self.read('A,BB\n')

    def test_invalid_ids_duplicate_rows_and_escaping_symlinks(self):
        (self.root/'A.nii').touch()
        for name in ('../A', '/A', 'sub/A', 'A.nii', 'A.nii.gz', '.', ''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.read(f'{name},BB\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.read('A,BB\nA,T1CE\n')
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside)/'outside.nii'
            target.touch()
            (self.root/'outside.nii').symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'outside input root'):
                self.read('outside,BB\n')

    def test_dicom_folder_and_legacy_pair_contracts(self):
        folder = self.root/'subject'/'series'
        folder.mkdir(parents=True)
        for mode in ('dicom', 'nifti-pairs'):
            self.assertEqual(self.read('folder,sequence\nsubject/series,BB\n', mode),
                             [('subject/series', 'BB', folder)])
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                self.read('subject/series,BB\nsubject/./series,T1CE\n', mode)

    def test_run_sh_reads_actual_nifti_files_and_records_ids(self):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        affine = np.diag([-0.5, 0.75, 2., 1.])
        for name in ('A.nii', 'B.nii.gz'):
            nib.save(nib.Nifti1Image(data, affine), self.root/name)
        self.csv.write_text('file_id,sequence\nA,T1CE\nB,BB\n')
        bundle = Path(__file__).resolve().parents[1]
        output = self.root/'output'
        result = subprocess.run(['bash', str(bundle/'run.sh'), '--input-format', 'nifti',
            '--input-root', str(self.root), '--csv', str(self.csv), '--output', str(output),
            '--validate-inputs-only'], env={**os.environ, 'PYTHON': sys.executable},
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        manifest = json.loads((output/'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'completed')
        self.assertEqual([r['file_id'] for r in manifest['volumes']], ['A', 'B'])
        for row in manifest['volumes']:
            self.assertEqual(row['shape'], list(data.shape))
            np.testing.assert_allclose(row['affine'], affine)
            self.assertNotIn('folder', row)
        self.assertFalse(list(output.rglob('*.nii*')))


if __name__ == '__main__':
    unittest.main()
