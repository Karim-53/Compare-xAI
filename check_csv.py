from pathlib import Path

import pandas as pd
from src.utils import root


def test_csv(root='./'):
  try:
    for path in Path(root).rglob('*.csv'):
      df = pd.read_csv(path, index_col=0)
      assert df.index.is_unique, path
  except Exception as e:
    print('\nCSV file causing the error', path)
    raise

test_csv(root)
