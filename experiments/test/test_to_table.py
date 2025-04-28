# Unit test function for creating LaTeX table

from pandas import DataFrame
import pytest

from experiments.latex import to_table


@pytest.fixture
def df1():
    return DataFrame({"Method": ["Method1", "Method2", "Method3"],
                      "Value": [0.81234, 0.61234, 0.91234]})


@pytest.fixture
def options():
    return {
        "caption": "Example Table",
        "label": "tab:example",
        "table_placement": "H",
        "column_widths": ["l", "c"],
        "decimals": 3
    }


BEGIN_TABLE = ("\n" +
               "\\begin{{table}}[{}]\n" +
               "\\centering\n" +
               "\\begin{{tabular}}{{{}}}\n" +
               "\\toprule\n")
END_TABLE = ("\\bottomrule\n" +
             "\\end{{tabular}}\n" +
             "\\caption{{{}}}\n" +
             "\\label{{{}}}\n" +
             "\\end{{table}}")


def expected(headers: tuple, rows: tuple, options: dict):
    """
        Used to construct what the generated LaTeX should be.

        :param tuple headers: header for each column
        :param tuple rows: of tuples of expected cell values
        :param dict options: table options

        :returns str: expected LaTeX
    """
    result = BEGIN_TABLE.format(options['table_placement'],
                                "".join(options['column_widths']))

    result += " & ".join(headers) + " \\\\\n"

    result += "\\midrule\n"

    for row in rows:
        result += " & ".join(row) + " \\\\\n"

    result += END_TABLE.format(options['caption'], options['label'])

    return result


def test_to_table_type_error_1_():  # no arguments specified
    with pytest.raises(TypeError):
        to_table()


def test_to_table_type_error_2_(options):  # df not a dataframe
    with pytest.raises(TypeError):
        to_table(df='invalid', options=options)


def test_to_table_value_error_1_(df1, options):  # unknown options key
    options['invalid'] = 'bad'
    with pytest.raises(ValueError):
        to_table(df1, options)


def test_to_table_value_error_2_(df1, options):  # invalid table_placement
    options["table_placement"] = "invalid"
    with pytest.raises(ValueError):
        to_table(df1, options)


def test_to_table_value_error_3_(df1):  # missing required options
    options = {"caption": "Missing Label"}
    with pytest.raises(ValueError):
        to_table(df1, options)


def test_to_table_ok_1_(df1):  # default options
    options = {
        "caption": "?",
        "label": "?"
    }

    latex_table = to_table(df1, options=options)
    assert latex_table == expected(('Method', 'Value'),
                                   (("Method1", "0.81"),
                                    ("Method2", "0.61"),
                                    ("Method3", "0.91")),
                                   {"caption": "?",
                                    "label": "?",
                                    "table_placement": "ht",
                                    "column_widths": ["l", "l"]})
    print(latex_table)


def test_to_table_ok_2_(df1, options):  # all options specified
    latex_table = to_table(df1, options)

    assert latex_table == expected(('Method', 'Value'),
                                   (("Method1", "0.812"),
                                    ("Method2", "0.612"),
                                    ("Method3", "0.912")),
                                   options)
    print(latex_table)


def test_to_table_ok_3_(df1):  # no column_widths specified
    options = {
        "caption": "No Column Widths",
        "label": "tab:no_widths",
        "table_placement": "H",
        "decimals": 2
    }

    latex_table = to_table(df1, options=options)
    assert latex_table == expected(('Method', 'Value'),
                                   (("Method1", "0.81"),
                                    ("Method2", "0.61"),
                                    ("Method3", "0.91")),
                                   {"caption": "No Column Widths",
                                    "label": "tab:no_widths",
                                    "table_placement": "H",
                                    "column_widths": ["l", "l"]})
    print(latex_table)


def test_to_table_ok_4_(df1):  # no decimals specified
    options = {
        "caption": "No Decimals",
        "label": "tab:no_decimals",
        "table_placement": "H",
        "column_widths": ["l", "c"]
    }

    latex_table = to_table(df1, options=options)
    assert latex_table == expected(('Method', 'Value'),
                                   (("Method1", "0.81"),
                                    ("Method2", "0.61"),
                                    ("Method3", "0.91")),
                                   {"caption": "No Decimals",
                                    "label": "tab:no_decimals",
                                    "table_placement": "H",
                                    "column_widths": ["l", "c"]})
    print(latex_table)
