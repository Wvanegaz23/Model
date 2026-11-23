import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluar_retiro import CATEGORICAL, NUMERIC, load_data, make_model


class EvaluacionTests(unittest.TestCase):
    def test_pipeline_handles_unseen_categories_and_missing_values(self):
        X = pd.DataFrame({
            **{name: [1.0, 2.0, None, 4.0] for name in NUMERIC},
            **{name: ["a", "b", "a", None] for name in CATEGORICAL},
        })
        model = make_model().fit(X, [0, 1, 0, 1])
        unseen = X.iloc[[0]].copy()
        unseen["segmento"] = "desconocido"
        self.assertEqual(model.predict_proba(unseen).shape, (1, 2))

    def test_duplicate_join_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"id": [1, 2], "Target": [0, 1]}).to_csv(
                root / "train.csv", sep=";", index=False
            )
            pd.DataFrame({"id": [1, 1]}).to_excel(
                root / "train_test_demograficas.xlsx", index=False
            )
            with self.assertRaises(pd.errors.MergeError):
                load_data(root)

    def test_invalid_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"id": [1, 2], "Target": [0, 2]}).to_csv(
                root / "train.csv", sep=";", index=False
            )
            for name in ("train_test_demograficas.xlsx", "train_test_subsidios.xlsx"):
                pd.DataFrame({"id": [1, 2]}).to_excel(root / name, index=False)
            with self.assertRaisesRegex(ValueError, "Target"):
                load_data(root)


if __name__ == "__main__":
    unittest.main()
