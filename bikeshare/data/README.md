# Data sources

- [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), Hadi Fanaee-T, DOI 10.24432/C5W894, CC BY 4.0. The file `rentals.csv` is the source archive's `day.csv`. The pipeline reads the date and total count.
- [NOAA GHCN-Daily](https://www.ncei.noaa.gov/pub/data/ghcn/daily/), station [USW00013743](https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USW00013743.dly), Washington Reagan National Airport. The file `weather.csv` contains the 2011–2012 daily minimum temperature, maximum temperature, and precipitation from that independently published station record. Extraction skips missing values and observations with nonblank quality flags. Source integers are divided by ten to obtain degrees Celsius and millimeters.

Both snapshots were retrieved on November 12, 2026. The observations cover the same 731 dates. The rental file also includes its own weather columns. The pipeline uses the NOAA observations as the weather features.
