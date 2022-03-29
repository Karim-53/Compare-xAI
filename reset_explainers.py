import pandas as pd
from sqlalchemy.dialects.mssql.information_schema import columns
import jinja2
from src.io import load_results, detail
from src.utils import root
# todo(s) from Tim <3
#   header row in specific color
#   one empty line before each test
#   merge redudent txt
#   another font in the table
#   fix \n in src
#   remove underscore (importance_symmetric) and put human readable info

EXPLAINERS_HTML_PATH = root + "/docs/explainers/"  # todo move to src.utils
def explainer_to_html(explainer_df_with_results):
    """ clean the df and write it to an html page"""
    df = explainer_df_with_results.fillna('')
    df = df.replace([{}], '')
    # os.chdir(file_dirname)

    # Generate HTML from template.
    template = jinja2.Template("""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width">
        <title>Demo</title>
        <link href="https://cdn.jsdelivr.net/npm/simple-datatables@latest/dist/style.css" rel="stylesheet" type="text/css">
        <script src="https://cdn.jsdelivr.net/npm/simple-datatables@latest" type="text/javascript"></script>
        </head>

        <body>
            {{ dataframe }}
        </body>

        <script defer type="text/javascript">
            let myTable = new simpleDatatables.DataTable("#myTable");
        </script>
    </html>"""
                               )

    output_html = template.render(dataframe=df.to_html(table_id="myTable"))

    # Write generated HTML to file.
    file = EXPLAINERS_HTML_PATH + df.loc[0, 'value'] + ".htm"
    print(file)
    with open(file, "w", encoding="utf-8") as file_obj:
        file_obj.write(output_html)


if __name__ == "__main__":
    from src.scoring import get_score_df, get_eligible_points_df, get_summary_df
    from explainers import Random

    result_df = load_results()
    result_df_detailed = detail(result_df)

    explainer = Random()
    explainer_df = pd.DataFrame(explainer.to_pandas())
    explainer_df.columns = ['value']
    explainer_df['test'] = explainer_df.index
    explainer_df['sub_test_category'] = None
    explainer_df['sub_test'] = None
    cols = ['test', 'sub_test_category', 'sub_test', explainer.name]
    result_s = result_df_detailed[cols]
    cols = ['test', 'sub_test_category', 'sub_test', 'value']
    result_s.columns = cols
    explainer_df_with_results = pd.concat([
        explainer_df,
        result_s,
    ], axis=0,).reset_index()[cols]
    print(explainer_df_with_results)
    explainer_df_with_results['remarks'] = None

    explainer_to_html(explainer_df_with_results)
