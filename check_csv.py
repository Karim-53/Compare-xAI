import pandas as pd
from src.utils import root


def test(root='./'):
  try:
    for path in Path(root).rglob('*.csv'):
      df = pd.read_csv(path)
      assert df.Index.is_unique; path
  except Exception as e:
    print('\nCSV file causing the error', path)
    raise
      
