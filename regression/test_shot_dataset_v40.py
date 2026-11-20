import csv
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.shot_dataset_v40 import (
    FIELDNAMES, evaluate_predictions, load_csv, manifest_from_reports,
    validate_manifest, write_csv,
)


def example_report(video='vk.mp4', frame=35):
    return {'video_name': video, 'source_video': {'total_frames': 127, 'fps': 25},
            'impact_events': [{'impact_frame': frame}]}


class ShotDatasetV40Tests(unittest.TestCase):
    def setUp(self):
        self.rows = manifest_from_reports([example_report('vk.mp4', 35),
                                           example_report('rohit.mp4', 123)])

    def reviewed(self, index, label, split):
        self.rows[index]['shot_label'] = label
        self.rows[index]['review_status'] = 'Reviewed'
        self.rows[index]['annotator'] = 'human-reviewer'
        self.rows[index]['split'] = split

    def test_report_import_never_infers_shot_type(self):
        self.assertEqual([r['shot_label'] for r in self.rows], ['Unlabeled', 'Unlabeled'])
        self.assertTrue(validate_manifest(self.rows)['valid'])

    def test_frame_windows_contain_impact(self):
        row = self.rows[0]
        self.assertEqual((row['start_frame'], row['impact_frame'], row['end_frame']), (15, 35, 51))

    def test_clamps_video_frame_range(self):
        rows = manifest_from_reports([example_report('short.mp4', 3)], 20, 200)
        self.assertEqual(rows[0]['start_frame'], 1)
        self.assertEqual(rows[0]['end_frame'], 127)

    def test_invalid_impact_outside_video(self):
        with self.assertRaises(ValueError):
            manifest_from_reports([example_report(frame=128)])

    def test_duplicate_clip_ids_rejected(self):
        self.rows.append(deepcopy(self.rows[0]))
        self.assertIn('duplicate clip_id', ' '.join(validate_manifest(self.rows)['errors']))

    def test_label_requires_review(self):
        self.rows[0]['shot_label'] = 'Drive'
        self.assertFalse(validate_manifest(self.rows)['valid'])

    def test_reviewed_must_have_label(self):
        self.rows[0]['review_status'] = 'Reviewed'
        self.assertFalse(validate_manifest(self.rows)['valid'])

    def test_only_known_labels_allowed(self):
        self.rows[0]['shot_label'] = 'Cover Drive'
        self.assertFalse(validate_manifest(self.rows)['valid'])

    def test_split_requires_reviewed_training_class(self):
        self.rows[0]['split'] = 'train'
        self.assertFalse(validate_manifest(self.rows)['valid'])

    def test_same_recording_cannot_cross_splits(self):
        self.reviewed(0, 'Drive', 'train')
        clone = deepcopy(self.rows[0])
        clone['clip_id'] = 'vk_second_clip'
        clone['shot_label'] = 'Cut'
        clone['split'] = 'test'
        self.rows.append(clone)
        self.assertIn('Data leakage', ' '.join(validate_manifest(self.rows)['errors']))

    def test_same_video_cannot_change_recording_id(self):
        clone = deepcopy(self.rows[0])
        clone['clip_id'] = 'vk_second_clip'
        clone['recording_id'] = 'vk_different'
        self.rows.append(clone)
        self.assertIn('Data leakage', ' '.join(validate_manifest(self.rows)['errors']))

    def test_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.csv'
            write_csv(path, self.rows)
            self.assertEqual(load_csv(path)[0]['clip_id'], self.rows[0]['clip_id'])
            self.assertTrue(validate_manifest(load_csv(path))['valid'])

    def test_no_labeled_test_is_not_fake_accuracy(self):
        result = evaluate_predictions(self.rows, [])
        self.assertIsNone(result['accuracy_all'])
        self.assertEqual(result['evaluated_clips'], 0)

    def test_evaluation_correct_and_abstain(self):
        self.reviewed(0, 'Drive', 'test')
        self.reviewed(1, 'Cut', 'test')
        result = evaluate_predictions(self.rows, [
            {'clip_id': self.rows[0]['clip_id'], 'predicted_label': 'Drive'},
            {'clip_id': self.rows[1]['clip_id'], 'predicted_label': 'Uncertain'},
        ])
        self.assertEqual(result['evaluated_clips'], 2)
        self.assertEqual(result['coverage'], .5)
        self.assertEqual(result['accuracy_all'], .5)
        self.assertEqual(result['accuracy_when_classified'], 1.0)
        self.assertEqual(result['per_class']['Cut']['recall'], 0)

    def test_duplicate_prediction_rejected(self):
        self.reviewed(0, 'Defense', 'test')
        pred = {'clip_id': self.rows[0]['clip_id'], 'predicted_label': 'Defense'}
        with self.assertRaises(ValueError):
            evaluate_predictions(self.rows, [pred, pred])

    def test_invalid_prediction_class_rejected(self):
        self.reviewed(0, 'Pull', 'test')
        with self.assertRaises(ValueError):
            evaluate_predictions(self.rows, [{'clip_id': self.rows[0]['clip_id'], 'predicted_label': 'Cover Drive'}])

    def test_missing_prediction_is_abstention(self):
        self.reviewed(0, 'Pull', 'test')
        result = evaluate_predictions(self.rows, [])
        self.assertEqual(result['classified_clips'], 0)
        self.assertEqual(result['abstained_clips'], 1)
        self.assertIsNone(result['accuracy_when_classified'])

    def test_source_video_path_must_be_plain_filename(self):
        self.rows[0]['source_video'] = '../test.mp4'
        self.assertFalse(validate_manifest(self.rows)['valid'])


if __name__ == '__main__':
    unittest.main()
