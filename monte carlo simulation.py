import pandas as pd
import numpy as np
from scipy.stats import beta, norm, lognorm
import warnings
warnings.filterwarnings('ignore')

# ===== 1. ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ =====
print("="*70)
print("ΔΙΟΡΘΩΜΕΝΗ ΔΗΜΙΟΥΡΓΙΑ ΣΕΝΑΡΙΩΝ MONTE CARLO")
print("="*70)

df = pd.read_csv(
    r"C:\Users\30697\Desktop\ΔΙΠΛΩΜΑΤΙΚΗ\PublicDataset\SampleProsumer_total.csv",
    sep=";"
)

# Μετατροπές
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Month'] = df['Datetime'].dt.month
df['Timestep'] = df.index % 96

print(f"✓ Φορτώθηκαν {len(df)} γραμμές")
print(f"✓ Περίοδος: {df['Datetime'].min().date()} έως {df['Datetime'].max().date()}")

# ===== 2. ΥΠΟΛΟΓΙΣΜΟΣ ΣΤΑΤΙΣΤΙΚΩΝ ΑΝΑ ΜΗΝΑ & TIMESTEP =====
# ΚΡΙΣΙΜΗ ΑΛΛΑΓΗ: Προσθήκη Month στο groupby
stats = df.groupby(['Month', 'Timestep']).agg({
    'Load': ['mean', 'std'],
    'Production': ['mean', 'std'],
    'DAM_Values': ['mean', 'std'],
    'Tariff_Charges': ['mean', 'std']
}).reset_index()

stats.columns = ['Month', 'Timestep',
                 'load_mean', 'load_std',
                 'production_mean', 'production_std',
                 'damvalues_mean', 'damvalues_std',
                 'tariffcharges_mean', 'tariffcharges_std']

print(f"✓ Στατιστικά για {stats['Month'].nunique()} μήνες x 96 timesteps")

# ===== 3. ΔΗΜΙΟΥΡΓΙΑ ΣΕΝΑΡΙΩΝ =====
T = 96  # timesteps ανά ημέρα
N = 10  # αριθμός σεναρίων
DAYS = 365

rows = []

print("\nΔημιουργία σεναρίων...")

for day in range(DAYS):
    # ΚΡΙΣΙΜΗ ΑΛΛΑΓΗ: Υπολογισμός μήνα για την ημέρα
    # Μετατροπή day (0-364) σε μήνα (1-12)
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cumsum = 0
    month = 1
    for m, days in enumerate(month_days, 1):
        cumsum += days
        if day < cumsum:
            month = m
            break
    
    PV_scenarios = np.zeros((N, T))
    Load_scenarios = np.zeros((N, T))
    BuyPrice_scenarios = np.zeros((N, T))
    SellPrice_scenarios = np.zeros((N, T))
    
    for t in range(T):
        # ΚΡΙΣΙΜΗ ΑΛΛΑΓΗ: Φιλτράρισμα στατιστικών για συγκεκριμένο μήνα
        month_t_stats = stats[(stats['Month'] == month) & (stats['Timestep'] == t)]
        
        if len(month_t_stats) == 0:
            # Fallback αν δεν υπάρχουν δεδομένα
            PV_scenarios[:, t] = 0.0
            Load_scenarios[:, t] = 0.1
            BuyPrice_scenarios[:, t] = 0.15
            SellPrice_scenarios[:, t] = 0.10
            continue
        
        # Λήψη τιμών
        mu_pv = month_t_stats['production_mean'].values[0]
        sigma_pv = month_t_stats['production_std'].values[0]
        mu_load = month_t_stats['load_mean'].values[0]
        sigma_load = month_t_stats['load_std'].values[0]
        mu_sell = month_t_stats['damvalues_mean'].values[0]
        sigma_sell = month_t_stats['damvalues_std'].values[0]
        mu_buy = month_t_stats['tariffcharges_mean'].values[0]
        sigma_buy = month_t_stats['tariffcharges_std'].values[0]
        
        # ===== ΠΑΡΑΓΩΓΗ PV =====
        # ΚΡΙΣΙΜΗ ΑΛΛΑΓΗ: Έλεγχος για νύχτα
        if mu_pv < 0.01:  # Νυχτερινή ώρα
            PV_scenarios[:, t] = 0.0  # ΣΚΛΗΡΟ ΜΗΔΕΝ!
        else:
            # Χρήση Beta κατανομής
            if sigma_pv > 0 and 0 < mu_pv < 1:
                var = sigma_pv**2
                if var < mu_pv * (1 - mu_pv):  # Έλεγχος εγκυρότητας
                    m = mu_pv * (mu_pv * (1 - mu_pv) / var - 1)
                    n = (1 - mu_pv) * (mu_pv * (1 - mu_pv) / var - 1)
                    
                    if m > 0 and n > 0:
                        # Δειγματοληψία από Beta
                        R = np.random.randn(N)
                        PV_scenarios[:, t] = beta.ppf(norm.cdf(R), m, n)
                    else:
                        # Fallback: Normal με clipping
                        PV_scenarios[:, t] = np.clip(
                            np.random.normal(mu_pv, sigma_pv, N),
                            0.0, 1.0
                        )
                else:
                    # Fallback: Normal με clipping
                    PV_scenarios[:, t] = np.clip(
                        np.random.normal(mu_pv, max(sigma_pv, 0.01), N),
                        0.0, 1.0
                    )
            else:
                PV_scenarios[:, t] = np.clip(
                    np.random.normal(mu_pv, max(sigma_pv, 0.01), N),
                    0.0, 1.0
                )
        
        # ===== ΚΑΤΑΝΑΛΩΣΗ (Lognormal) =====
        if mu_load > 0 and sigma_load > 0:
            mu_ln = np.log(mu_load**2 / np.sqrt(sigma_load**2 + mu_load**2))
            sigma_ln = np.sqrt(np.log(1 + (sigma_load**2 / mu_load**2)))
            
            x = np.random.randn(N)
            Load_scenarios[:, t] = np.exp(mu_ln + sigma_ln * x)
        else:
            # Fallback: Normal με clipping
            Load_scenarios[:, t] = np.clip(
                np.random.normal(max(mu_load, 0.01), max(sigma_load, 0.01), N),
                0.0, None
            )
        
        # ===== ΤΙΜΕΣ ΕΝΕΡΓΕΙΑΣ (Normal) =====
        # SellPrice
        if sigma_sell > 0:
            x_sell = np.random.randn(N)
            SellPrice_scenarios[:, t] = mu_sell + sigma_sell * x_sell
        else:
            SellPrice_scenarios[:, t] = mu_sell
        
        # BuyPrice
        if sigma_buy > 0:
            x_buy = np.random.randn(N)
            BuyPrice_scenarios[:, t] = mu_buy + sigma_buy * x_buy
        else:
            BuyPrice_scenarios[:, t] = mu_buy
    
    # Αποθήκευση σε λίστα
    for s in range(N):
        for t in range(T):
            global_timestep = day * T + t
            rows.append({
                'timestep': global_timestep,
                'Day': day,
                'scenario': s,
                'PV': max(PV_scenarios[s, t], 0.0),  # Ensure non-negative
                'Load': max(Load_scenarios[s, t], 0.0),
                'BuyPrice': BuyPrice_scenarios[s, t],
                'SellPrice': SellPrice_scenarios[s, t]
            })
    
    # Progress
    if (day + 1) % 50 == 0:
        print(f"  Ολοκληρώθηκαν {day + 1}/{DAYS} ημέρες...")

print(f"\n✓ Δημιουργήθηκαν {len(rows)} γραμμές")

# ===== 4. ΑΠΟΘΗΚΕΥΣΗ =====
df_scenarios = pd.DataFrame(rows)

output_file = r'C:\Users\30697\Desktop\ΔΙΠΛΩΜΑΤΙΚΗ\monte_carlo_scenarios.xlsx'
df_scenarios.to_excel(output_file, index=False)

print(f"✓ Αποθηκεύτηκε: monte_carlo_scenarios.xlsx")

# ===== 5. VALIDATION =====
print("\n" + "="*70)
print("VALIDATION - ΣΥΓΚΡΙΣΗ ΣΤΑΤΙΣΤΙΚΩΝ")
print("="*70)

# Σύγκριση
orig_pv_mean = df['Production'].mean()
scen_pv_mean = df_scenarios['PV'].mean()
orig_pv_std = df['Production'].std()
scen_pv_std = df_scenarios['PV'].std()

orig_load_mean = df['Load'].mean()
scen_load_mean = df_scenarios['Load'].mean()
orig_load_std = df['Load'].std()
scen_load_std = df_scenarios['Load'].std()

print("\nΠΑΡΑΓΩΓΗ PV:")
print(f"  Original - Mean: {orig_pv_mean:.4f} kW, Std: {orig_pv_std:.4f} kW")
print(f"  Scenarios - Mean: {scen_pv_mean:.4f} kW, Std: {scen_pv_std:.4f} kW")
print(f"  Διαφορά Mean: {abs(scen_pv_mean - orig_pv_mean):.4f} kW")

print("\nΚΑΤΑΝΑΛΩΣΗ:")
print(f"  Original - Mean: {orig_load_mean:.4f} kW, Std: {orig_load_std:.4f} kW")
print(f"  Scenarios - Mean: {scen_load_mean:.4f} kW, Std: {scen_load_std:.4f} kW")
print(f"  Διαφορά Mean: {abs(scen_load_mean - orig_load_mean):.4f} kW")

# Έλεγχος νυχτερινής PV
night_timesteps = [0, 1, 2, 3, 4, 5, 20, 21, 22, 23]
night_mask = df_scenarios['timestep'].apply(lambda x: (x % 96) in night_timesteps)
night_pv_mean = df_scenarios[night_mask]['PV'].mean()

print(f"\nΝΥΧΤΕΡΙΝΗ PV (timesteps {night_timesteps}):")
print(f"  Μέση τιμή: {night_pv_mean:.6f} kW")
if night_pv_mean < 0.001:
    print(f"  ✓ ΣΩΣΤΟ: Μηδενική νυχτερινή παραγωγή")
else:
    print(f"  ⚠️ ΠΡΟΣΟΧΗ: Υπάρχει νυχτερινή παραγωγή!")

# Εποχιακότητα
df_scenarios['Month'] = df_scenarios['Day'].apply(lambda d: 
    next((m for m, (start, end) in enumerate([
        (0, 31), (31, 59), (59, 90), (90, 120), (120, 151),
        (151, 181), (181, 212), (212, 243), (243, 273),
        (273, 304), (304, 334), (334, 365)
    ], 1) if start <= d < end), 12)
)

monthly_pv = df_scenarios.groupby('Month')['PV'].mean()

print("\nΕΠΟΧΙΑΚΟΤΗΤΑ PV (Μηνιαίοι μέσοι όροι):")
for month in range(1, 13):
    if month in monthly_pv.index:
        print(f"  Μήνας {month:2d}: {monthly_pv[month]:.4f} kW")

print("\n" + "="*70)
print("✓ ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ")
print("="*70)
print(f"\nΑρχείο: {output_file}")

