import logging
import sys

import numpy as np
import numpy as np
import pandas as pd

import symfit as sf
from symfit.core.minimizers import BFGS, DifferentialEvolution


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Load data
logger.info("Loading data")
df = pd.read_csv(
    "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv/",
    parse_dates=["dateRep"],
    infer_datetime_format=True,
    dayfirst=True,
)

logger.info("Cleaning data")
df = df.rename(
    columns={"dateRep": "date", "countriesAndTerritories": "country"}
)  # Sane column names
df = df.drop(["day", "month", "year", "geoId"], axis=1)  # Not required

# Create DF with sorted index
sorted_data = df.set_index(df["date"]).sort_index()

# Remove all rows with zero deaths in it
sorted_data = sorted_data[sorted_data["deaths"] != 0]

sorted_data["cumulative_cases"] = sorted_data.groupby(by="country")["cases"].cumsum()
sorted_data["cumulative_deaths"] = sorted_data.groupby(by="country")["deaths"].cumsum()

# Filter out data with less than 100 deaths, we probably can't get very good estimates from these.
sorted_data = sorted_data[sorted_data["cumulative_deaths"] >= 100]

# Remove "Czechia" it has a population of NaN
sorted_data = sorted_data[sorted_data["country"] != "Czechia"]

# Get final list of countries
countries = sorted_data["country"].unique()
n_countries = len(countries)

# Pull out population size per country
populations = {
    country: df[df["country"] == country].iloc[0]["popData2018"]
    for country in countries
}

# A map from country to integer index (for the model)
idx_country = pd.Index(countries).get_indexer(sorted_data.country)

# Create a new column with the number of days since first infection (the x-axis)
country_first_dates = {
    c: sorted_data[sorted_data["country"] == c].index.min() for c in countries
}
sorted_data["100_cases"] = sorted_data.apply(
    lambda x: country_first_dates[x.country], axis=1
)
sorted_data["days_since_100_cases"] = (
    sorted_data.index - sorted_data["100_cases"]
).apply(lambda x: x.days)

logger.info("Training...")
fit_result = {}
ode_model = {}
for country in countries:
    t, S, I, R, D = sf.variables("t, S, I, R, D")
    p_susceptible = 0.00085
    N_mu = populations[country] * p_susceptible
    β_0, γ_0, μ_0 = 0.35, 0.1, 0.03
    N_0 = N_mu
    β = sf.Parameter("β", value=β_0, min=0.1, max=0.5)
    γ = sf.Parameter("γ", value=γ_0, min=0.01, max=0.2)
    N = sf.Parameter("N", value=N_0, min=1e4, max=1e7)
    μ = sf.Parameter("μ", value=μ_0, min=0.0001, max=0.1)

    print(country, N_0)

    model_dict = {
        sf.D(S, t): -β * I * S / N,
        sf.D(I, t): β * I * S / N - γ * I - μ * I,
        sf.D(R, t): γ * I,
        sf.D(D, t): μ * I,
    }

    p_infected = 0.01
    I_0, R_0, D_0 = N_mu * p_infected, N_mu * p_infected - 100.0, 100.0
    S_0 = N_mu - I_0 - R_0 - D_0
    ode_model[country] = sf.ODEModel(
        model_dict, initial={t: 0.0, S: S_0, I: I_0, R: R_0, D: D_0}
    )

    idx = sorted_data["country"] == country
    x = sorted_data[idx]["days_since_100_cases"].values
    y = sorted_data[idx]["cumulative_deaths"].values

    fit = sf.Fit(
        ode_model[country],
        t=x,
        S=None,
        I=None,
        R=None,
        D=y,
        minimizer=[DifferentialEvolution, BFGS],
    )
    fit_result[country] = fit.execute(
        DifferentialEvolution={"seed": 0, "tol": 1e-2, "maxiter": 5}, BFGS={"tol": 1e-6}
    )
    print(fit_result[country])

logger.info("Inferencing...")
n_days = 365  # Daily predictions

cumulative_prediction = {}
daily_prediction = {}
residuals_high = {}
residuals_low = {}
for country in countries:
    idx = sorted_data["country"] == country
    x = sorted_data[idx]["days_since_100_cases"].values
    y = sorted_data[idx]["cumulative_deaths"].values
    tvec = np.arange(x.max() + n_days)
    d, i, r, s = ode_model[country](t=tvec, **fit_result[country].params)
    cumulative_prediction[country] = d
    y = sorted_data[idx]["deaths"].values
    daily_prediction[country] = np.diff(d)
    residual_std = np.std(y - daily_prediction[country][: len(y)])
    residuals_high[country] = daily_prediction[country] + residual_std
    residuals_low[country] = daily_prediction[country] - residual_std

# Remember this is one big vector that contains all countries at all times.
# To do inference we need to construct a new vector with new times
# Create the time index
time_index = np.arange(0, n_days, 1)
time_index = np.repeat(time_index, n_countries)

# Create the country index
country_index = np.arange(n_countries)
country_index = np.tile(country_index, n_days)
dummy_y = np.zeros(len(time_index))

logger.info("Saving model")

# Calculate dates (must be in python datetime to work with pydantic)
country_start = [country_first_dates[x] for x in countries[country_index].tolist()]
country_offset = [pd.DateOffset(x) for x in time_index]
dates = list(
    map(lambda x: (x[0] + x[1]).to_pydatetime(), zip(country_start, country_offset))
)

# Create a big dataframe with all this info
predictions = pd.DataFrame(
    {
        "timestamp": dates,
        "country": countries[country_index],
        "deaths_prediction": [
            daily_prediction[c][t]
            for t, c in zip(time_index, countries[country_index].tolist())
        ],
        "cumulative_deaths_prediction": [
            cumulative_prediction[c][t]
            for t, c in zip(time_index, countries[country_index].tolist())
        ],
        "residuals_low": [
            residuals_low[c][t]
            for t, c in zip(time_index, countries[country_index].tolist())
        ],
        "residuals_high": [
            residuals_high[c][t]
            for t, c in zip(time_index, countries[country_index].tolist())
        ],
        "days_since_100_cases": time_index,
    },
    index=dates,
)

# Merge in the ground truth
predictions = pd.merge(
    predictions.rename_axis("index").reset_index(),
    sorted_data[["country", "deaths", "cumulative_deaths"]]
    .rename_axis("index")
    .reset_index(),
    on=["index", "country"],
    how="outer",
).set_index("index")

# Save to file
predictions.to_pickle("predictions.pkl")
logger.info("Finished")
