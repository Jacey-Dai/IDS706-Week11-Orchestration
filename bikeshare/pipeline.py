"""Join historical daily rental totals with independently recorded airport weather."""
from pathlib import Path
import os

import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUTPUT = ROOT / "results"


def clean(source):
    if source == "rentals":
        frame = pd.read_csv(DATA / "rentals.csv", usecols=["dteday", "cnt"])
        frame = frame.rename(columns={"dteday": "date", "cnt": "rentals"})
        if (frame.rentals < 0).any():
            raise ValueError("Negative rental count")
    elif source == "weather":
        frame = pd.read_csv(DATA / "weather.csv")
        if ((frame.tmin_c > frame.tmax_c) | (frame.rain_mm < 0)).any():
            raise ValueError("Invalid weather observation")
    else:
        raise ValueError(f"Unknown source: {source}")
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
    frame = frame.drop_duplicates().sort_values("date")
    if frame.date.duplicated().any() or frame.isna().any().any():
        raise ValueError(f"Conflicting dates or missing observations in {source}")
    return frame


def load(rentals, weather):
    joined = rentals.merge(weather, on="date", how="left", validate="one_to_one")
    if joined.isna().any().any():
        raise ValueError("Rental day has no matching weather observation")
    engine = create_engine(os.environ["BIKE_DATABASE_URL"])
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE IF NOT EXISTS daily_bikes (date DATE PRIMARY KEY, rentals INTEGER NOT NULL, tmin_c DOUBLE PRECISION NOT NULL, tmax_c DOUBLE PRECISION NOT NULL, rain_mm DOUBLE PRECISION NOT NULL)"))
        connection.execute(text("DELETE FROM daily_bikes"))
        connection.execute(text("INSERT INTO daily_bikes VALUES (CAST(:date AS DATE), :rentals, :tmin_c, :tmax_c, :rain_mm)"), joined.to_dict("records"))
    return len(joined)


def analyze():
    engine = create_engine(os.environ["BIKE_DATABASE_URL"])
    with engine.connect() as connection:
        daily = pd.read_sql_query(text("SELECT * FROM daily_bikes ORDER BY date"), connection)
        summary = pd.read_sql_query(text("""
            SELECT CASE WHEN rain_mm > 0 THEN 'wet' ELSE 'dry' END AS weather,
                   count(*) AS days, round(avg(rentals), 1) AS mean_rentals
            FROM daily_bikes GROUP BY 1 ORDER BY 1
        """), connection)
    OUTPUT.mkdir(exist_ok=True)
    daily.to_csv(OUTPUT / "daily.csv", index=False)
    summary.to_csv(OUTPUT / "weather_summary.csv", index=False)
    return len(daily)
