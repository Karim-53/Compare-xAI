from typing import Iterable

import jinja2
import pandas as pd

from src.explainer import valid_explainers
from src.io import detail
from src.dask.utils import load_results
from src.utils import root

# todo(s) from Tim <3
#   header row in specific color
#   one empty line before each test
#   merge redudent txt
#   another font in the table
#   fix \n in src
#   remove underscore (importance_symmetric) and put human readable info
# todo [after acceptance] page for each paper: searchi ken el paper saye 9rineh w 7atineh wala mahouch yet considered, ( enehom unit tests / xai / original results eli fih)

EXPLAINERS_HTML_PATH = root + "/docs/explainers/"  # todo move to src.utils

import re

_urlfinderregex = re.compile(r'http([^<\.\s]+\.[^<\.\s]*)+[^<\.\s]{2,}')  # todo fix tr at the end of the link


def linkify(text, maxlinklength=256):
    def replacewithlink(matchobj):
        url = matchobj.group(0)
        text = str(url)
        if text.startswith('http://'):
            text = text.replace('http://', '', 1)
        elif text.startswith('https://'):
            text = text.replace('https://', '', 1)

        if text.startswith('www.'):
            text = text.replace('www.', '', 1)

        if len(text) > maxlinklength:
            halflength = maxlinklength / 2
            text = text[0:halflength] + '...' + text[len(text) - halflength:]

        return '<a class="comurl" href="' + url + '" target="_blank" rel="nofollow">' + text + '<img class="imglink" src="./images/linkout.png"></a>'

    if text != None and text != '':
        return _urlfinderregex.sub(replacewithlink, text)
    else:
        return ''


def explainer_to_html(explainer_df_with_results):
    """ clean the df and write it to an html page"""
    df = explainer_df_with_results.fillna('')
    df = df.applymap(lambda x: '\n '.join(x) if isinstance(x, Iterable) and not isinstance(x, str) else x)
    # df['value'] = df['value'].apply(lambda x: linkify(x) if isinstance(x, str) else x)
    # os.chdir(file_dirname)

    # Generate HTML from template.
    template = jinja2.Template("""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">

        <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">

        <title>""" + df.loc[0, 'value'] + """</title>
        <link href="https://cdn.jsdelivr.net/npm/simple-datatables@latest/dist/style.css" rel="stylesheet" type="text/css">
        <script src="https://cdn.jsdelivr.net/npm/simple-datatables@latest" type="text/javascript"></script>
        
        <style>
            body {
              margin: 0;
              font-family: Arial, Helvetica, sans-serif;
            }
            
            .topnav {
              overflow: hidden;
              background-color: #333;
            }
            
            .topnav a {
              float: left;
              color: #f2f2f2;
              text-align: center;
              padding: 14px 16px;
              text-decoration: none;
              font-size: 17px;
            }
            
            .topnav a:hover {
              background-color: #ddd;
              color: black;
            }
            
            .topnav a.active {
              background-color: #04AA6D;
              color: white;
            }
            </style>

        </head>

        <body>
            <div style="display: flex; width: 100%; height: 50px; flex-direction: column; background-color: gray; overflow: hidden;">
                <iframe src="../header.html"  style="flex-grow: 1; border: none; margin: 0; padding: 0;" ></iframe>
            </div>
            {{ dataframe }}
        </body>

        <script defer type="text/javascript">
            let myTable = new simpleDatatables.DataTable("#myTable");
        </script>
        <style type="text/css">
        .imglink {
          width: 18px;
          padding-left: 3px;
        }
        </style>
    </html>"""
                               )

    table_html = df.to_html(table_id="myTable")
    table_html = linkify(table_html)
    output_html = template.render(dataframe=table_html)
    output_html = output_html.replace(r'\n', ' <br> ')
    # Write generated HTML to file.
    file = EXPLAINERS_HTML_PATH + df.loc[0, 'value'] + ".htm"
    print(file)
    with open(file, "w", encoding="utf-8") as file_obj:
        file_obj.write(output_html)


if __name__ == "__main__":
    result_df = load_results()
    result_df_detailed = detail(result_df)

    for explainer_class in valid_explainers:
        print(explainer_class.name)
        # explainer = explainer_class()
        explainer_df = pd.DataFrame(explainer_class.to_pandas())
        explainer_df.columns = ['value']
        explainer_df['test'] = explainer_df.index
        explainer_df['sub_test_category'] = None
        explainer_df['sub_test'] = None
        cols = ['test', 'sub_test_category', 'sub_test', explainer_class.name]
        result_s = result_df_detailed[cols]
        cols = ['test', 'sub_test_category', 'sub_test', 'value']
        result_s.columns = cols
        explainer_df_with_results = pd.concat([
            explainer_df,
            result_s,
        ], axis=0, ).reset_index()[cols]
        # print(explainer_df_with_results)
        explainer_df_with_results['remarks'] = None
        # todo also add columns results from original paper , and in remarks we can write why they are not the same

        explainer_to_html(explainer_df_with_results)

    for explainer_class in valid_explainers:
        # 'https://karim-53.github.io/Compare-xAI/explainers/'
        url = explainer_class.name + '.htm'
        print(f'<a href="{url}">{explainer_class.name}</a><br>')
    print('End')
