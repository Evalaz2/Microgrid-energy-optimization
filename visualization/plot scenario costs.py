import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, norm, lognorm

# Διάβασε τα δεδομένα
df_results = pd.read_excel("stochastic_optimization_results.xlsx")
df = pd.read_excel(r"C:\Users\30697\Desktop\ΔΙΠΛΩΜΑΤΙΚΗ\monte_carlo_scenarios10.xlsx")

num_scenarios = df['scenario'].nunique()
DAY_STEPS = 96

if "Day" not in df.columns:
    df["Day"] = (df["timestep"] // DAY_STEPS).astype(int)

print(f"Σενάρια: {num_scenarios}")
print(f"Ημέρες στα αποτελέσματα: {df_results['Day'].nunique()}")

# Έλεγχος αν υπάρχει η στήλη Scenario_Probability
if 'Scenario_Probability' in df_results.columns:
    print("\n✓ Βρέθηκε η στήλη 'Scenario_Probability' στο Excel")
    
    # Παίρνουμε τις πιθανότητες ανά σενάριο
    scenario_probs = df_results.groupby('Scenario')['Scenario_Probability'].first()
    
    print("\n=== ΠΙΘΑΝΟΤΗΤΕΣ ΣΕΝΑΡΙΩΝ (από Excel) ===")
    for s, p in scenario_probs.items():
        print(f"Σενάριο {s}: Πιθανότητα = {p:.6f} ({p*100:.2f}%)")
    
    # Έλεγχος αν αθροίζουν σε 1
    total_prob = scenario_probs.sum()
    print(f"\nΆθροισμα πιθανοτήτων: {total_prob:.6f}")
    if abs(total_prob - 1.0) > 1e-6:
        print("⚠️ ΠΡΟΣΟΧΗ: Οι πιθανότητες ΔΕΝ αθροίζουν σε 1!")
    else:
        print("✓ Οι πιθανότητες αθροίζουν σωστά σε 1")
        
else:
    print("\n⚠️ ΣΦΑΛΜΑ: Δεν βρέθηκε η στήλη 'Scenario_Probability' στο Excel!")
    print("Πρόσθεσε πρώτα την πιθανότητα στο TWO-STAGE-SO-2.py")
    exit()

# ========== ΥΠΟΛΟΓΙΣΜΟΣ ΣΤΑΘΜΙΣΜΕΝΟΥ ΜΕΣΟΥ ΚΟΣΤΟΥΣ ==========
deltat = 0.25

# Υπολογισμός κόστους για κάθε timestep
df_results['Cost'] = (df_results['Total_Pg_import'] * df_results['BuyPrice'] * deltat - 
                      df_results['Total_Pg_export'] * df_results['SellPrice'] * deltat)



# Σταθμισμένος μέσος όρος ανά ημέρα
def weighted_mean(group):
    return np.average(group['Cost'], weights=group['Scenario_Probability'])

daily_weighted_cost = df_results.groupby('Day').apply(weighted_mean).reset_index()
daily_weighted_cost.columns = ['Day', 'WeightedMeanCost']

# Απλός μέσος για σύγκριση
daily_simple_mean = df_results.groupby('Day')['Cost'].mean().reset_index()
daily_simple_mean.columns = ['Day', 'SimpleMeanCost']

# ========== ΔΙΑΓΡΑΜΜΑ ==========
fig, ax = plt.subplots(figsize=(20, 8))

# Σταθμισμένος μέσος όρος
ax.plot(daily_weighted_cost['Day'], daily_weighted_cost['WeightedMeanCost'], 
        color='#0066CC', linewidth=2.5, marker='o', markersize=4, 
        label='Σταθμισμένος Μέσος Όρος (με πιθανότητες)', alpha=0.9, zorder=3)


# Οριζόντιες γραμμές
ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)

weighted_mean_total = daily_weighted_cost['WeightedMeanCost'].mean()
ax.axhline(y=weighted_mean_total, color='red', linestyle=':', linewidth=2, 
           label=f'Μέσος Όρος: {weighted_mean_total:.3f} €/ημέρα')

# Ρυθμίσεις
ax.set_xlabel("Ημέρα", fontsize=14, fontweight='bold')
ax.set_ylabel("Κόστος (€)", fontsize=14, fontweight='bold')
ax.set_title("Σταθμισμένο Μέσο Κόστος ανά Ημέρα - Two-Stage Stochastic\n(Με Πιθανότητες Σεναρίων)", 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.6)
ax.legend(fontsize=12, loc='best', framealpha=0.95)
ax.tick_params(axis='both', labelsize=12)

# Ρύθμιση X axis
total_days = len(daily_weighted_cost)
if total_days > 50:
    ax.set_xticks(daily_weighted_cost['Day'][::5])
elif total_days > 30:
    ax.set_xticks(daily_weighted_cost['Day'][::3])

plt.tight_layout()
plt.savefig('σταθμισμενο_κοστος_σωστο.png', dpi=300, bbox_inches='tight')
plt.show()

# ΣΤΑΤΙΣΤΙΚΑ
print("\n=== ΣΤΑΤΙΣΤΙΚΑ ΚΟΣΤΟΥΣ ===")
print(f"\nΣταθμισμένος Μέσος Όρος (ΣΩΣΤΟΣ):")
print(f"  Μέσο ημερήσιο κόστος: {daily_weighted_cost['WeightedMeanCost'].mean():.3f} €")
print(f"  Συνολικό κόστος: {daily_weighted_cost['WeightedMeanCost'].sum():.2f} €")

diff = abs(daily_weighted_cost['WeightedMeanCost'].sum() - daily_simple_mean['SimpleMeanCost'].sum())
print(f"\nΔιαφορά: {diff:.2f} €")

print("\n✅ Διάγραμμα με σταθμισμένο μέσο όρο δημιουργήθηκε επιτυχώς!")