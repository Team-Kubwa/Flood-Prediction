from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("flood_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

raw_features = [
 'MonsoonIntensity','TopographyDrainage','RiverManagement',
 'Deforestation','Urbanization','ClimateChange','DamsQuality',
 'Siltation','AgriculturalPractices','Encroachments',
 'IneffectiveDisasterPreparedness','DrainageSystems',
 'CoastalVulnerability','Landslides','Watersheds',
 'DeterioratingInfrastructure','PopulationScore',
 'WetlandLoss','InadequatePlanning','PoliticalFactors'
]

def create_new_features(df, cols):
    df = df.copy()

    df['sum'] = df[cols].sum(axis=1)
    df['mean'] = df[cols].mean(axis=1)
    df['median'] = df[cols].median(axis=1)
    df['max'] = df[cols].max(axis=1)
    df['min'] = df[cols].min(axis=1)
    df['std'] = df[cols].std(axis=1)

    df['cov'] = df['std'] / df['mean'].replace(0, 1e-6)
    df['p25'] = df[cols].quantile(0.25, axis=1)
    df['p75'] = df[cols].quantile(0.75, axis=1)
    df['range'] = df['max'] - df['min']

    return df


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    df = df[raw_features]
    df = create_new_features(df, raw_features)
    df = df[feature_columns]

    prediction = model.predict(df)[0]

    return {
        "FloodProbability": float(prediction)
    }