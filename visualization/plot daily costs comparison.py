import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========== ΔΙΑΒΑΣΜΑ ΔΕΔΟΜΕΝΩΝ ==========

# 1. Two-Stage Stochastic Optimization
df_stochastic = pd.read_excel("stochastic_optimization_results.xlsx")

deltat = 0.25  # 15 λεπτά = 0.25 ώρες

# Υπολογισμός κόστους για Stochastic
df_stochastic['Cost'] = (df_stochastic['Total_Pg_import'] * df_stochastic['BuyPrice'] * deltat - 
                         df_stochastic['Total_Pg_export'] * df_stochastic['SellPrice'] * deltat)

# Μέσο κόστος ανά ημέρα (μέσος όρος από όλα τα σενάρια)
cost_per_scenario = df_stochastic.groupby(['Day', 'Scenario'])['Cost'].sum().reset_index()

# Βήμα 2: Προσθήκη πιθανοτήτων
if 'Scenario_Probability' in df_stochastic.columns:
    scenario_probs = df_stochastic.groupby('Scenario')['Scenario_Probability'].first()
    cost_per_scenario['Probability'] = cost_per_scenario['Scenario'].map(scenario_probs)
    print("✓ Χρήση πιθανοτήτων από το Excel")
else:
    print("⚠️ ΠΡΟΣΟΧΗ: Δεν βρέθηκε η στήλη 'Scenario_Probability'!")
    print("  Χρήση ομοιόμορφων πιθανοτήτων (1/num_scenarios)")
    num_scenarios = df_stochastic['Scenario'].nunique()
    cost_per_scenario['Probability'] = 1.0 / num_scenarios

# Βήμα 3: Σταθμισμένος μέσος όρος ανά ημέρα
cost_stochastic_daily = cost_per_scenario.groupby('Day').apply(
    lambda x: (x['Cost'] * x['Probability']).sum()
).reset_index()
cost_stochastic_daily.columns = ['Day', 'Cost_Stochastic']

print(f"\nStochastic - Ημέρες: {len(cost_stochastic_daily)}, Σενάρια: {df_stochastic['Scenario'].nunique()}")
# 2. MILP Deterministic
df_milp = pd.read_excel("first_stage_2.xlsx")

# Υπολογισμός κόστους για MILP
df_milp['Cost'] = (df_milp['Pg_import'] * df_milp['BuyPrice'] * deltat - 
                   df_milp['Pg_export'] * df_milp['SellPrice'] * deltat)

# Κόστος ανά ημέρα
cost_milp_daily = df_milp.groupby('Day')['Cost'].sum().reset_index()
cost_milp_daily.columns = ['Day', 'Cost_MILP']

print(f"Stochastic - Ημέρες: {len(cost_stochastic_daily)}, Σενάρια: {df_stochastic['Scenario'].nunique()}")
print(f"MILP - Ημέρες: {len(cost_milp_daily)}")

# ========== ΣΥΓΚΡΙΣΗ ΚΟΣΤΟΥΣ ==========
fig, ax = plt.subplots(figsize=(20, 8))

# Γραμμή για Stochastic
ax.plot(cost_stochastic_daily['Day'], cost_stochastic_daily['Cost_Stochastic'], 
        color='#0066CC', linewidth=2.5, marker='o', markersize=4, 
        label='Two-Stage Stochastic (Μέσος Όρος Σεναρίων)', alpha=0.85)

# Γραμμή για MILP
ax.plot(cost_milp_daily['Day'], cost_milp_daily['Cost_MILP'], 
        color='#DC143C', linewidth=2.5, marker='s', markersize=4, 
        label='MILP Deterministic', alpha=0.85)

# Οριζόντια γραμμή στο 0 (break-even)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Ρυθμίσεις
ax.set_xlabel("Ημέρα", fontsize=14, fontweight='bold')
ax.set_ylabel("Κόστος (€)", fontsize=14, fontweight='bold')
ax.set_title("Σύγκριση Καθημερινού Κόστους: Two-Stage Stochastic vs MILP", 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.6)
ax.legend(fontsize=12, loc='best', framealpha=0.95)
ax.tick_params(axis='both', labelsize=12)

# Ρύθμιση X axis
total_days = max(len(cost_stochastic_daily), len(cost_milp_daily))
if total_days > 50:
    step = 5
elif total_days > 30:
    step = 3
else:
    step = 1

all_days = sorted(set(cost_stochastic_daily['Day'].tolist() + cost_milp_daily['Day'].tolist()))
ax.set_xticks(all_days[::step])

plt.tight_layout()
plt.savefig('συγκριση_κοστους_Stochastic_vs_MILP.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== ΣΤΑΤΙΣΤΙΚΑ ==========
print("\n=== ΣΤΑΤΙΣΤΙΚΑ ΚΟΣΤΟΥΣ ===")
print(f"\nTwo-Stage Stochastic:")
print(f"  Μέσο ημερήσιο κόστος: {cost_stochastic_daily['Cost_Stochastic'].mean():.3f} €")
print(f"  Συνολικό κόστος: {cost_stochastic_daily['Cost_Stochastic'].sum():.2f} €")
print(f"  Τυπική απόκλιση: {cost_stochastic_daily['Cost_Stochastic'].std():.3f} €")
print(f"  Min: {cost_stochastic_daily['Cost_Stochastic'].min():.3f} €")
print(f"  Max: {cost_stochastic_daily['Cost_Stochastic'].max():.3f} €")

print(f"\nMILP Deterministic:")
print(f"  Μέσο ημερήσιο κόστος: {cost_milp_daily['Cost_MILP'].mean():.3f} €")
print(f"  Συνολικό κόστος: {cost_milp_daily['Cost_MILP'].sum():.2f} €")
print(f"  Τυπική απόκλιση: {cost_milp_daily['Cost_MILP'].std():.3f} €")
print(f"  Min: {cost_milp_daily['Cost_MILP'].min():.3f} €")
print(f"  Max: {cost_milp_daily['Cost_MILP'].max():.3f} €")

# Σύγκριση
diff = cost_milp_daily['Cost_MILP'].sum() - cost_stochastic_daily['Cost_Stochastic'].sum()
print(f"\nΔιαφορά κόστους (MILP - Stochastic): {diff:.2f} €")
if diff < 0:
    print(f"Το MILP είναι φθηνότερο κατά {abs(diff):.2f} € ({abs(diff/cost_stochastic_daily['Cost_Stochastic'].sum())*100:.1f}%)")
else:
    print(f"Το Stochastic είναι φθηνότερο κατά {abs(diff):.2f} € ({abs(diff/cost_milp_daily['Cost_MILP'].sum())*100:.1f}%)")