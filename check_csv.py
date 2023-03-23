import pandas as pd
from src.utils import root

def test(root):
  try:
    for path in Path(constants.root).rglob('*.csv'):
      df = pd.read_csv(path)
      print(df.Index.is_unique)
      except Exception as e:
        print('\nCSV file causing the error', path)
        raise
      
