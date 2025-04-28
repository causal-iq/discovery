
# Generate LaTex representation of tables etc.

from pandas import DataFrame
from pydantic import BaseModel, Field, model_validator, ValidationError, \
    ConfigDict
from typing import List, Optional


class TableOptions(BaseModel):
    """
        Options for customizing the LaTeX table.

        :param str caption: caption for the table (optional).
        :param str label: label for referencing the table in LaTeX (optional).
        :param str table_placement: table placement specifier (e.g. 'ht', 'H').
        :param list column_widths: list of column widths or alignments
            (e.g., ['l', 'c', 'r']).
        :param int decimals: number of decimal places to round numeric values
            (default is 2).
    """
    caption: str  # Mandatory field
    label: str  # Mandatory field
    table_placement: str = "ht"
    column_widths: Optional[List[str]] = None
    decimals: int = 2
    df_columns: List[str] = Field(default_factory=list)  # Used for validation

    # Pydantic configuration to forbid extra fields
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def validate(cls, **data):
        def format_dict_human_readable(d: dict) -> str:
            return "\n".join(f"{key}: {value}" for key, value in d.items())

        try:
            return cls(**data)
        except ValidationError as e:
            # Safeguard against empty or unexpected 'loc'
            error_messages = format_dict_human_readable(
                {f"Field '{err['loc'][0] if err['loc'] else 'unknown'}'":
                 err["msg"] for err in e.errors()})
            raise ValueError(f"Invalid options:\n{error_messages}") from e

    @model_validator(mode="after")
    def validate_model(cls, values):
        # Default to left-aligned columns if column_widths is not provided
        if values.column_widths is None:
            values.column_widths = ["l"] * len(values.df_columns)
        return values

    @model_validator(mode="after")
    def validate_table_placement(cls, values):
        if values.table_placement not in ["ht", "H", "b", "t", "p"]:
            raise ValueError("Invalid table placement specifier")
        return values


def to_table(df: DataFrame, options: dict = {}) -> str:
    """
        Convert a pandas DataFrame to a LaTeX tabular representation compatible
        with the booktabs package.

        :param DataFrame df: pandas DataFrame to convert.
        :param dict options: dictionary of options for customizing the table.

        :return: LaTeX tabular string.
    """
    if not isinstance(df, DataFrame) or not isinstance(options, dict):
        raise TypeError('to_table() bad arg type')

    # Validate and convert the dictionary into a TableOptions object
    options["df_columns"] = df.columns.tolist()
    options = TableOptions.validate(**options)
    print(options.column_widths)

    # Round numeric values in the DataFrame
    df = df.copy()  # Avoid modifying the original DataFrame
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].round(options.decimals)

    # Start building the LaTeX string
    latex_str = f"\n\\begin{{table}}[{options.table_placement}]\n"
    latex_str += "\\centering\n"
    latex_str += "\\begin{tabular}{" + "".join(options.column_widths) + "}\n"
    latex_str += "\\toprule\n"

    # Add column headers
    latex_str += " & ".join(df.columns) + " \\\\\n"
    latex_str += "\\midrule\n"

    # Add rows
    for _, row in df.iterrows():
        latex_str += " & ".join(map(str, row)) + " \\\\\n"

    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"

    # Add caption and label if provided
    if options.caption:
        latex_str += f"\\caption{{{options.caption}}}\n"
    if options.label:
        latex_str += f"\\label{{{options.label}}}\n"

    latex_str += "\\end{table}"

    return latex_str
