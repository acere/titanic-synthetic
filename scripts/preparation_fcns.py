import pandas as pd
from pandas.api.types import is_numeric_dtype


def explicit_na(df, col_name, inplace=False):
    """ Deal with the NaN for categorical and numerical columns

    for categorical replaces NaN with string `na`
    for numerical, replaces NaN with the median and add a bool column marking
    the NaN values
    """
    col = df[col_name]
    df_new = df[[col_name]]
    def_value = "na"

    if is_numeric_dtype(col):
        def_value = col.median()
        if col_name in df.select_dtypes("integer").columns:
            def_value = int(def_value)
        df_new.loc[:, col_name + "_na"] = col.isna()
    df_new.loc[:, col_name] = col.fillna(def_value, downcast="infer")

    if inplace:
        df[df_new.columns] = df_new
        return
    return df_new


def type_conversion(df: pd.DataFrame, type_from: str, type_to: str):
    """ Converts all columns of type `type_from` to
    `type_tp`
    """
    df[df.select_dtypes(type_from).columns] = df.select_dtypes(type_from).astype(
        type_to
    )


def prepare_dict(df: pd.DataFrame):
    """ Build schema of dataframe in JSON format

    Infer column format from the data type.preparation_fcns.py
    #TODO: automate index regexp
    """

    fields_int = [
        dict(name=k, subtype="integer", type="number", uniques=0)
        for k in list(df.select_dtypes("int"))
    ]

    fields_float = [
        dict(name=k, subtype="float", type="number", uniques=0)
        for k in list(df.select_dtypes("float"))
    ]

    fields_cat = [
        dict(name=k, subtype="categorical", type="categorical", uniques=df[k].nunique())
        for k in list(df.select_dtypes(["object", "category"]))
    ]

    fields_bool = [
        dict(name=k, subtype="boolean", type="categorical", uniques=2)
        for k in list(df.select_dtypes("bool"))
    ]
    primary = dict(
        name="PassengerId",
        subtype="integer",
        type="number",
        uniques=0,
        regex="^[0-9]{6}$",
    )
    return fields_float + fields_cat + fields_bool + fields_int + [primary]


def process_dataset(f_name, labels=True):
    cols = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Cabin",
        "Ticket",
    ]

    if labels:
        cols += ["Survived"]

    df = pd.read_csv(f_name, index_col=0)
    df = df[cols]

    # process cabine text
    cabin_split = (
        df["Cabin"].str.split().str[0].str.split(r"([A-Z])", expand=True).loc[:, 1:]
    )
    cabin_split[2] = pd.to_numeric(cabin_split[2], downcast="unsigned").astype("Int64")
    cabin_split.rename({1: "cabin_txt", 2: "cabin_num"}, axis="columns", inplace=True)
    df[cabin_split.columns] = cabin_split

    cabin_count = df["Cabin"].str.split().str.len().astype("Int64")
    df["cabin_count"] = cabin_count

    df.drop("Cabin", axis=1, inplace=True)

    # process Ticket text
    ticket_num = pd.to_numeric(
        df.Ticket.str.replace(".", "").str.split().str[-1], errors="coerce"
    ).astype("Int64")
    ticket_txt = (
        df.Ticket.str.replace(".", "")
        .str.rsplit(n=1)
        .str[:-1]
        .str[0]
        .str.replace(" ", "")
    )
    df["ticket_txt"] = ticket_txt
    df["ticket_num"] = ticket_num
    df.drop("Ticket", axis=1, inplace=True)

    if labels:
        df["Survived"] = df["Survived"].astype("bool")

    col_cat = df.columns[df.nunique().between(3, 50)]
    df[col_cat] = df[col_cat].astype("object")

    cols_na = list(df.columns[df.isna().any()])
    [explicit_na(df, k, True) for k in cols_na]

    return df


def best_metric(data, metric, pretty=True, dataset="test"):
    mean = data[f"{dataset}-{metric}-mean"].max()
    std = data[f"{dataset}-{metric}-std"][data[f"{dataset}-{metric}-mean"].idxmax()]
    if pretty:
        print("{}: {:.3f} +\- {:.3f}".format(metric, mean, std))
    return mean, std
