import re
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# --- Daten einlesen und parsen ---
filename = "../data/medium-output.txt"
data_list = []

with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Verwende einen regulären Ausdruck, um Projekt, Jahr und die Werte zu extrahieren
        match = re.match(r'out\[(.*?)\]\[(.*?)\]\s*=\s*(.*)', line)
        if match:
            project = match.group(1)
            year = match.group(2)
            values_str = match.group(3)
            values = values_str.split(';')
            if len(values) >= 5:
                try:
                    bs_klassen   = float(values[0])
                    total_klassen = float(values[1])
                    bs_methoden  = float(values[2])
                    total_methoden = float(values[3])
                    autoren      = float(values[4])
                    
                    # Berechnung der Density für Klassen und Methoden separat
                    density_classes = bs_klassen / total_klassen if total_klassen > 0 else np.nan
                    density_methods = bs_methoden / total_methoden if total_methoden > 0 else np.nan
                    
                    data_list.append({
                        "project": project,
                        "year": int(year),
                        "bs_klassen": bs_klassen,
                        "total_klassen": total_klassen,
                        "bs_methoden": bs_methoden,
                        "total_methoden": total_methoden,
                        "autoren": autoren,
                        "density_classes": density_classes,
                        "density_methods": density_methods
                    })
                except Exception as e:
                    print("Fehler bei der Umwandlung der Werte in Zeile:", line, "\n", e)
            else:
                print("Nicht genügend Werte in Zeile:", line)
        else:
            print("Kein passendes Format in Zeile:", line)

# DataFrame erstellen
df = pd.DataFrame(data_list)
print("Erste Zeilen der eingelesenen Daten:")
print(df.head())

# --- Between-Subject Analyse ---
# Aggregieren auf Projektebene: Durchschnittswerte für Autoren und beide Density-Felder
agg = df.groupby("project").agg({
    "autoren": "mean",
    "density_classes": "mean",
    "density_methods": "mean"
}).reset_index()

print("\nAggregierte Daten pro Projekt:")
print(agg.head())

print("\n--- Between-Subject Analyse ---")

# Für density_classes
print("\n*** Analysen für density_classes ***")
corr_cls, p_value_cls = stats.pearsonr(agg["autoren"], agg["density_classes"])
print(f"Pearson-Korrelation (Klassen): {corr_cls:.3f}, p-Wert: {p_value_cls:.3f}")
model_between_classes = smf.ols("density_classes ~ autoren", data=agg).fit()
print(model_between_classes.summary())

# Für density_methods
print("\n*** Analysen für density_methods ***")
corr_meth, p_value_meth = stats.pearsonr(agg["autoren"], agg["density_methods"])
print(f"Pearson-Korrelation (Methoden): {corr_meth:.3f}, p-Wert: {p_value_meth:.3f}")
model_between_methods = smf.ols("density_methods ~ autoren", data=agg).fit()
print(model_between_methods.summary())

# --- Within-Subject Analyse ---
print("\n--- Within-Subject Analyse ---")

# Für density_classes
print("\n*** Mixed-Effects Modell für density_classes ***")
model_within_classes = smf.mixedlm("density_classes ~ autoren", data=df, groups=df["project"]).fit()
print(model_within_classes.summary())

# Für density_methods
print("\n*** Mixed-Effects Modell für density_methods ***")
model_within_methods = smf.mixedlm("density_methods ~ autoren", data=df, groups=df["project"]).fit()
print(model_within_methods.summary())
