# ============================================================
#  Maharashtra Rainfall Analysis — Agri-tech Portfolio Project 1
#  Run this entirely FREE on Google Colab: colab.research.google.com
#  No downloads needed. All data fetched automatically.
# ============================================================

# ── STEP 0: Install libraries (run this cell first in Colab) ──────────────
# Uncomment the line below only if running in Colab
# !pip install requests pandas matplotlib seaborn prophet --quiet


# ── STEP 1: Import libraries ──────────────────────────────────────────────
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

print("✅ Libraries loaded successfully!")


# ── STEP 2: Fetch real rainfall data from NASA POWER API ─────────────────
# NASA POWER is 100% free, no API key required.
# We pull monthly rainfall (PRECTOTCORR) for Maharashtra's centre point.
# Latitude 19.75°N, Longitude 75.71°E  →  heart of Maharashtra

def fetch_nasa_rainfall(lat=19.75, lon=75.71, start_year=2010, end_year=2023):
    """
    Fetches monthly corrected total precipitation from NASA POWER API.
    Returns a cleaned pandas DataFrame with Year, Month, Rainfall_mm columns.
    """
    url = (
        "https://power.larc.nasa.gov/api/temporal/monthly/point"
        f"?parameters=PRECTOTCORR"
        f"&community=AG"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start_year}&end={end_year}"
        "&format=JSON"
    )

    print(f"📡 Fetching rainfall data from NASA POWER ({start_year}–{end_year})...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    monthly_data = data["properties"]["parameter"]["PRECTOTCORR"]

    records = []
    for key, value in monthly_data.items():
        year = int(key[:4])
        month = int(key[4:])
        records.append({"Year": year, "Month": month, "Rainfall_mm": value})

    df = pd.DataFrame(records).sort_values(["Year", "Month"]).reset_index(drop=True)

    # Replace fill values (-999) with NaN
    df["Rainfall_mm"] = df["Rainfall_mm"].replace(-999.0, np.nan)

    print(f"✅ Data fetched: {len(df)} monthly records ({start_year}–{end_year})")
    return df


df = fetch_nasa_rainfall()
print("\nFirst 5 rows of raw data:")
print(df.head())


# ── STEP 3: Engineer useful features ─────────────────────────────────────

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
KHARIF_MONTHS  = [6, 7, 8, 9, 10]   # June–October  (main sowing season)
RABI_MONTHS    = [11, 12, 1, 2, 3]  # Nov–March     (winter crop)

df["Month_Name"] = df["Month"].map(MONTH_NAMES)
df["Season"] = df["Month"].apply(
    lambda m: "Kharif" if m in KHARIF_MONTHS
    else ("Rabi" if m in RABI_MONTHS else "Summer")
)

# Annual totals
annual = (
    df.groupby("Year")["Rainfall_mm"]
    .sum()
    .reset_index()
    .rename(columns={"Rainfall_mm": "Annual_mm"})
)

# Kharif season totals (most critical for soybean, cotton, jowar)
kharif = (
    df[df["Month"].isin(KHARIF_MONTHS)]
    .groupby("Year")["Rainfall_mm"]
    .sum()
    .reset_index()
    .rename(columns={"Rainfall_mm": "Kharif_mm"})
)

annual = annual.merge(kharif, on="Year")

# Long-term averages and anomaly
lta_annual = annual["Annual_mm"].mean()
lta_kharif = annual["Kharif_mm"].mean()
annual["Annual_Anomaly_pct"] = ((annual["Annual_mm"] - lta_annual) / lta_annual * 100).round(1)
annual["Drought_Flag"] = annual["Annual_Anomaly_pct"] < -20   # >20% below normal = drought

print("\n📊 Annual Summary Table:")
print(annual.to_string(index=False))


# ── STEP 4: Monthly average — which months get the most rain? ─────────────
monthly_avg = (
    df.groupby(["Month", "Month_Name"])["Rainfall_mm"]
    .mean()
    .reset_index()
    .sort_values("Month")
)


# ── STEP 5: Visualisations ────────────────────────────────────────────────
# Set a clean style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Maharashtra Rainfall Analysis  |  Portfolio Project — Agri-tech Climate Analyst",
             fontsize=15, fontweight="bold", y=1.01)

# ── Chart 1: Annual rainfall trend ───────────────────────────────────────
ax1 = axes[0, 0]
colors = ["#d62728" if v < -20 else "#1f77b4" for v in annual["Annual_Anomaly_pct"]]
ax1.bar(annual["Year"], annual["Annual_mm"], color=colors, width=0.7, edgecolor="white")
ax1.axhline(lta_annual, color="orange", linewidth=1.8, linestyle="--", label=f"LTA {lta_annual:.0f} mm")
ax1.set_title("Annual Rainfall (mm)", fontweight="bold")
ax1.set_xlabel("Year"); ax1.set_ylabel("Rainfall (mm)")
ax1.legend()
# Annotate drought years
for _, row in annual[annual["Drought_Flag"]].iterrows():
    ax1.annotate("Drought", xy=(row["Year"], row["Annual_mm"]),
                 xytext=(0, 8), textcoords="offset points",
                 ha="center", fontsize=8, color="#d62728", fontweight="bold")

# ── Chart 2: Kharif season rainfall (June–October) ───────────────────────
ax2 = axes[0, 1]
kharif_colors = ["#d62728" if v < lta_kharif * 0.80 else "#2ca02c" for v in annual["Kharif_mm"]]
ax2.bar(annual["Year"], annual["Kharif_mm"], color=kharif_colors, width=0.7, edgecolor="white")
ax2.axhline(lta_kharif, color="orange", linewidth=1.8, linestyle="--", label=f"LTA {lta_kharif:.0f} mm")
ax2.set_title("Kharif Season Rainfall  (Jun–Oct)", fontweight="bold")
ax2.set_xlabel("Year"); ax2.set_ylabel("Rainfall (mm)")
ax2.legend()

# ── Chart 3: Monthly average distribution ────────────────────────────────
ax3 = axes[1, 0]
bar_colors = ["#2ca02c" if m in KHARIF_MONTHS else
              "#1f77b4" if m in RABI_MONTHS else "#ff7f0e"
              for m in monthly_avg["Month"]]
ax3.bar(monthly_avg["Month_Name"], monthly_avg["Rainfall_mm"],
        color=bar_colors, width=0.7, edgecolor="white")
ax3.set_title("Average Monthly Rainfall (all years)", fontweight="bold")
ax3.set_xlabel("Month"); ax3.set_ylabel("Avg Rainfall (mm)")
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#2ca02c", label="Kharif (Jun–Oct)"),
                   Patch(facecolor="#1f77b4", label="Rabi (Nov–Mar)"),
                   Patch(facecolor="#ff7f0e", label="Summer")]
ax3.legend(handles=legend_elements, fontsize=9)

# ── Chart 4: Annual anomaly % heatmap style ───────────────────────────────
ax4 = axes[1, 1]
anomaly_colors = ["#d62728" if v < -10 else "#2ca02c" if v > 10 else "#aec7e8"
                  for v in annual["Annual_Anomaly_pct"]]
bars = ax4.barh(annual["Year"].astype(str), annual["Annual_Anomaly_pct"],
                color=anomaly_colors, edgecolor="white")
ax4.axvline(0, color="black", linewidth=0.8)
ax4.set_title("Annual Rainfall Anomaly (%)\nvs. Long-Term Average", fontweight="bold")
ax4.set_xlabel("Anomaly (%)")
ax4.set_ylabel("Year")
# Add value labels
for bar, val in zip(bars, annual["Annual_Anomaly_pct"]):
    ax4.text(val + (1 if val >= 0 else -1), bar.get_y() + bar.get_height()/2,
             f"{val:+.1f}%", va="center", ha="left" if val >= 0 else "right",
             fontsize=8)

plt.tight_layout()
plt.savefig("maharashtra_rainfall_analysis.png", dpi=150, bbox_inches="tight")
print("\n✅ Chart saved as: maharashtra_rainfall_analysis.png")
plt.show()


# ── STEP 6: Key Findings Report (auto-generated) ─────────────────────────

drought_years  = annual[annual["Drought_Flag"]]["Year"].tolist()
best_year      = annual.loc[annual["Annual_mm"].idxmax(), "Year"]
worst_year     = annual.loc[annual["Annual_mm"].idxmin(), "Year"]
peak_month     = monthly_avg.loc[monthly_avg["Rainfall_mm"].idxmax(), "Month_Name"]
kharif_share   = (annual["Kharif_mm"].mean() / annual["Annual_mm"].mean() * 100)

print("\n" + "="*60)
print("  KEY FINDINGS — MAHARASHTRA RAINFALL ANALYSIS")
print("="*60)
print(f"  Period analysed     : {annual['Year'].min()} – {annual['Year'].max()}")
print(f"  Long-term avg (LTA) : {lta_annual:.0f} mm/year")
print(f"  Best rainfall year  : {best_year}  ({annual.loc[annual['Year']==best_year,'Annual_mm'].values[0]:.0f} mm)")
print(f"  Worst rainfall year : {worst_year} ({annual.loc[annual['Year']==worst_year,'Annual_mm'].values[0]:.0f} mm)")
print(f"  Drought years (>20% below LTA): {drought_years if drought_years else 'None in this period'}")
print(f"  Peak rainfall month : {peak_month}")
print(f"  Kharif share of annual rain  : {kharif_share:.1f}%")
print("="*60)
print("\n📌 AGRONOMIC INSIGHTS:")
print(f"  • {kharif_share:.0f}% of annual rainfall falls in the kharif season —")
print("    soybean, cotton, and jowar crops depend almost entirely on monsoon.")
if drought_years:
    print(f"  • Drought years identified: {drought_years}")
    print("    → These are high crop-failure risk years for insurance analysis.")
print(f"  • Weakest monsoon month to watch: "
      f"{monthly_avg.sort_values('Rainfall_mm').iloc[0]['Month_Name']} "
      f"({monthly_avg.sort_values('Rainfall_mm').iloc[0]['Rainfall_mm']:.1f} mm avg)")
print("\n✅ Analysis complete! Share this on LinkedIn as your first portfolio post.")


# ── STEP 7 (BONUS): Simple 2-year rainfall forecast using Prophet ─────────
# Uncomment the block below after running Steps 0–6 successfully.

# from prophet import Prophet
#
# # Prepare data in Prophet format (ds = date, y = value)
# prophet_df = annual[["Year", "Annual_mm"]].copy()
# prophet_df["ds"] = pd.to_datetime(prophet_df["Year"].astype(str) + "-06-01")
# prophet_df = prophet_df.rename(columns={"Annual_mm": "y"})
#
# model = Prophet(yearly_seasonality=False, weekly_seasonality=False,
#                 daily_seasonality=False, uncertainty_samples=300)
# model.fit(prophet_df)
#
# future = model.make_future_dataframe(periods=2, freq="YE")
# forecast = model.predict(future)
#
# print("\n📈 2-Year Rainfall Forecast:")
# print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(3).to_string(index=False))
# model.plot(forecast)
# plt.title("Maharashtra Annual Rainfall Forecast (Prophet)")
# plt.savefig("rainfall_forecast.png", dpi=150, bbox_inches="tight")
# plt.show()
