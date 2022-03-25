import os
import pathlib

root = ''
for root_relative_dir in [r'./', r'../', r'../../', r'../../../']:
    if all([pathlib.Path(os.path.abspath(root_relative_dir + subdir)).exists() for subdir in
            ['explainers/', 'tests/']]):
        root = os.path.abspath(root_relative_dir)
        break
else:
    print('Unable to find the root dir')


def plot_tree(xgb_model, filename, rankdir='UT', num_trees=0):
    """
    Plot the tree in high resolution
    :param xgb_model: xgboost trained model
    :param filename: the pdf file where this is saved
    :param rankdir: direction of the tree: default Top-Down (UT), accepts:'LR' for left-to-right tree
    :return:
    """
    # todo [after submission] move to src/d07_visualization
    import xgboost as xgb
    import os
    gvz = xgb.to_graphviz(xgb_model, num_trees=num_trees, rankdir=rankdir)
    _, file_extension = os.path.splitext(filename)
    format = file_extension.strip('.').lower()
    data = gvz.pipe(format=format)
    with open(filename, 'wb') as f:
        f.write(data)


def get_importance(attribution):
    return abs(attribution).mean(axis=0)

