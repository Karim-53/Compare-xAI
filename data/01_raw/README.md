
`test.csv`: list of unit tests (implemented/ might be implemented / will never be implemented and why).
 - dataset: name of the dataset
 - dataset_source (paper, direct link)
 - dataset_size (integer)
 - model (xgboost, Neural networks, black-box custom function, etc.)
 - test_source_paper (paper, direct link)
 - test_implementation_link
 - test_metric: how to translate the end-user requirement into a score between 0 (failing) and 1 (correctly fullfiling the requirement)

`xai.csv`:  list of xAI        (implemented/ might be implemented / will never be implemented and why)

`score.csv`: cross table (xai-test) with scores reported from related work. if the score has to be normalized to [0, 1] then we report the exact score in `note.csv`

`note.csv`: cross table (xai-test) with some note reported from related work.

`paper.csv`: list of papers considered. so we don't analyse a paper twice.

todo create these files :))))

