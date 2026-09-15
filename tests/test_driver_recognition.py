import unittest

import numpy as np

from src.driver_recognition import DriverRecognizer


class DriverRecognizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = DriverRecognizer()

    def test_models_initialize(self):
        self.assertIsNotNone(self.recognizer.face_detector)
        self.assertIsNotNone(self.recognizer.face_recognizer)

    def test_cosine_comparison_and_threshold_decision(self):
        embedding = np.ones((1, 128), dtype=np.float32)

        similarity = self.recognizer.compare_embeddings(embedding, embedding)
        result = self.recognizer.recognize(embedding, {"driver_1": embedding})

        self.assertAlmostEqual(similarity, 1.0, places=5)
        self.assertEqual(result.driver_id, "driver_1")
        self.assertTrue(result.recognized)

    def test_invalid_frame_is_handled(self):
        result = self.recognizer.extract_embedding(np.array([], dtype=np.uint8))

        self.assertEqual(result.status, "invalid_frame")
        self.assertIsNone(result.embedding)


if __name__ == "__main__":
    unittest.main()
