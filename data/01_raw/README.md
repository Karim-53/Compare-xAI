
# `test.csv`
list of tests (implemented/ might be implemented / will never be implemented and why).

 - **test**: Id of the test
 - **category**: Fidelity, Fragility, Stability, Simplicity, Stress test
 - **is_shortlisted**: [0 or 1] does it respect the selection protocol (see paper)
 - **is_implemented**: [0 or 1]
 - **short_description**: one sentence description of the test
 - **description**: longer description
 - **why_not_shortlisted**
 - **dataset**: name of the dataset
 - **dataset_source** (paper, direct link)
 - **dataset_size** (integer)
 - **model** (xgboost, Neural networks, black-box custom function, etc.)
 - **test_source_paper** (paper, direct link)
 - **test_source_code**: link to the original code if imported
 - **test_implementation_link**: link in our repo to the implementation
 - **Input_type**
 - **test_metric**: how to translate the end-user requirement into a score between 0 (failing) and 1 (correctly fullfiling the requirement)
 - **Solution**: tips to make most of the xai successed in fulfilling this end-user requirement

# `xai.csv`
list of xAI        (implemented/ might be implemented / will never be implemented and why)

# `score.csv`
cross table (xai-test) with scores reported from related work. if the score has to be normalized to [0, 1] then we report the exact score in `note.csv`

# `note.csv`
cross table (xai-test) with some note reported from related work. todo create remaining file

# `paper.csv`
list of papers considered. so we don't analyse a paper twice.



