import os

import pandas as pd

base_folder = os.path.dirname(__file__)
input_folder = os.path.join(base_folder, "input")
output_folder = os.path.join(base_folder, "output")


def preprocess_icu_occupancy():
    """icu_occupancy_*.csv contains the germany wide used icu cases.
    It contains the number of occupied beds. From apprx. june on it also contains the number of newly admitted patients.
    It is derived from https://github.com/robert-koch-institut/Intensivkapazitaeten_und_COVID-19-Intensivbettenbelegung_in_Deutschland/blob/main/Intensivregister_Bundeslaender_Kapazitaeten.csv
    """

    input_file = os.path.join(
        input_folder, "Intensivregister_Bundeslaender_Kapazitaeten.csv"
    )

    df_icu = pd.read_csv(input_file, parse_dates=["datum"])
    df_icu = df_icu[["datum", "faelle_covid_aktuell", "faelle_covid_erstaufnahmen"]]
    df_icu.columns = ["datum", "icu_occupancy", "icu_admissions"]
    df_icu = df_icu.groupby("datum").sum()

    # Fill up the time from beginning of 2020 to data begin
    date_range = pd.date_range(
        start="2020-01-01", end=df_icu.index.min(), inclusive="left"
    )
    new_data = pd.DataFrame(0, index=date_range, columns=df_icu.columns)
    df_icu = pd.concat([new_data, df_icu])

    os.makedirs(output_folder, exist_ok=True)
    df_icu.to_csv(os.path.join(output_folder, "icu.csv"))

    return df_icu


if __name__ == "__main__":
    print("Preprocessing ICU occupancy data...")
    preprocess_icu_occupancy()
    print("Preprocessing done.")
