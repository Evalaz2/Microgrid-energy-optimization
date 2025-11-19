import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========== ΔΙΑΒΑΣΜΑ ΔΕΔΟΜΕΝΩΝ ==========

# 1. Stochastic Optimization
df_stochastic = pd.read_excel("stochastic_optimization_results.xlsx")

# Παίρνουμε τις πιθανότητες
scenario_probs = df_stochastic.groupby('Scenario')['Scenario_Probability'].first()
print(f"\nΠιθανότητες σεναρίων: {scenario_probs.values}")

# ΣΤΑΘΜΙΣΜΕΝΟΣ ΜΕΣΟΣ ΟΡΟΣ ΑΝΑ TIMESTEP
energy_flows = ['Total_Ppv_d', 'Total_Ppv_b', 'Total_Ppv_g',
                'Total_Pb_d', 'Total_Pb_g', 'Total_Pb_c',
                'Total_Pg_d', 'Total_Pg_b', 'Total_Pdisc',
                'Total_Pg_import', 'Total_Pg_export', 'SOC',
                'BuyPrice', 'SellPrice']

weighted_avg_data = []
for (day, timestep), group in df_stochastic.groupby(['Day', 'Timestep_in_Day']):
    row_data = {'Day': day, 'Timestep_in_Day': timestep}
    for flow in energy_flows:
        # Σταθμισμένος: x₀×p₀ + x₁×p₁ + ... + x₉×p₉
        weighted_mean = (group[flow] * group['Scenario_Probability']).sum()
        row_data[flow] = weighted_mean
    weighted_avg_data.append(row_data)

avg_per_timestep_sto = pd.DataFrame(weighted_avg_data)

# Μετατροπή σε kWh
energy_columns = ['Total_Ppv_d', 'Total_Ppv_b', 'Total_Ppv_g', 'Total_Pb_d', 'Total_Pb_g',
                  'Total_Pb_c', 'Total_Pg_d', 'Total_Pg_b', 'Total_Pdisc',
                  'Total_Pg_import', 'Total_Pg_export']
for col in energy_columns:
    avg_per_timestep_sto[col] = avg_per_timestep_sto[col] / 1000

# 2. MILP
df_milp = pd.read_excel("first_stage_2.xlsx")

# Μετατροπή σε kWh
energy_columns_milp = ['Ppv_d', 'Ppv_b', 'Ppv_g', 'Pb_d', 'Pb_g', 'Pb_c',
                       'Pg_d', 'Pg_b', 'Pdisc', 'Pg_import', 'Pg_export']
for col in energy_columns_milp:
    df_milp[col] = df_milp[col] / 1000

# ========== ΕΠΙΛΟΓΗ ΗΜΕΡΑΣ ==========
SELECTED_DAY = 130  # 

# Φιλτράρισμα δεδομένων για τη συγκεκριμένη ημέρα
daily_stochastic = avg_per_timestep_sto[avg_per_timestep_sto['Day'] == SELECTED_DAY].copy()
daily_milp = df_milp[df_milp['Day'] == SELECTED_DAY].copy()

# ΔΗΜΙΟΥΡΓΙΑ Timestep_in_Day ΓΙΑ MILP (αν δεν υπάρχει)
if 'Timestep_in_Day' not in daily_milp.columns:
    # Υπολογίζουμε το Timestep_in_Day από τη στήλη Timestep
    if 'Timestep' in daily_milp.columns:
        # Βρίσκουμε το πρώτο timestep της ημέρας
        min_timestep = daily_milp['Timestep'].min()
        # Timestep_in_Day = Timestep - min_timestep + 1
        daily_milp['Timestep_in_Day'] = daily_milp['Timestep'] - min_timestep + 1
    else:
        # Αν δεν υπάρχει Timestep, δημιουργούμε αύξοντα αριθμό
        daily_milp['Timestep_in_Day'] = range(1, len(daily_milp) + 1)

if len(daily_stochastic) == 0:
    print(f"❌ Δεν βρέθηκαν δεδομένα για την ημέρα {SELECTED_DAY}")
    exit()

print(f"\n📊 Δημιουργία διαγράμματος για ημέρα {SELECTED_DAY}")
print(f"   - Timesteps Stochastic: {len(daily_stochastic)}")
print(f"   - Timesteps MILP: {len(daily_milp)}")

# ========== ΟΡΙΣΜΟΣ ΣΤΥΛ ==========
flow_styles = {
    'PV→Demand': ('#FF6B35', '-', 2.5),
    'PV→Battery': ('#F7931E', '--', 2.2),
    'PV→Grid': ('#FDB813', '-.', 2.0),
    'Grid Import': ('#C1121F', '-', 2.3),
    'Grid Export': ('#780000', '--', 2.0),
    'Battery Charge': ('#004E89', '-', 2.2),
    'Battery Discharge': ('#1A659E', '--', 2.2),
    'SOC': ('#2A9D8F', '-', 2.5),
    'Buy Price': ('#E63946', ':', 2.0),
    'Sell Price': ('#06A77D', ':', 2.0),
}

# ========== ΔΗΜΙΟΥΡΓΙΑ ΔΙΑΓΡΑΜΜΑΤΟΣ ==========
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(18, 14))

# ===== SUBPLOT 1: STOCHASTIC =====
ax2 = ax1.twinx()

# Ροές ενέργειας
color, style, width = flow_styles['PV→Demand']
ax1.plot(daily_stochastic["Timestep_in_Day"], daily_stochastic["Total_Ppv_d"],
         label='PV → Ζήτηση', color=color, linestyle=style, linewidth=width, alpha=0.85)

color, style, width = flow_styles['PV→Battery']
ax1.plot(daily_stochastic["Timestep_in_Day"], daily_stochastic["Total_Ppv_b"],
         label='PV → Μπαταρία', color=color, linestyle=style, linewidth=width, alpha=0.85)

color, style, width = flow_styles['PV→Grid']
ax1.plot(daily_stochastic["Timestep_in_Day"], daily_stochastic["Total_Ppv_g"],
         label='PV → Δίκτυο', color=color, linestyle=style, linewidth=width, alpha=0.85)

color, style, width = flow_styles['Grid Import']
ax1.plot(daily_stochastic["Timestep_in_Day"], daily_stochastic["Total_Pg_import"],
         label='Αγορά από Δίκτυο', color=color, linestyle=style, linewidth=width, alpha=0.85)

color, style, width = flow_styles['Grid Export']
ax1.plot(daily_stochastic["Timestep_in_Day"], daily_stochastic["Total_Pg_export"],
         label='Πώληση σε Δίκτυο', color=color, linestyle=style, linewidth=width, alpha=0.85)

color, style, width = flow_styles['Battery Charge']
ax1.plot(daily_stochastic["Timestep_in_Day"], daily_stochastic["Total_Pb_c"],
         label='Φόρτιση Μπαταρίας', color=color, linestyle=style, linewidth=width, alpha=0.85)

color, style, width = flow_styles['Battery Discharge']
ax1.plot(daily_stochastic["Timestep_in_Day"], daily_stochastic["Total_Pdisc"],
         label='Εκφόρτιση Μπαταρίας', color=color, linestyle=style, linewidth=width, alpha=0.85)

# SOC και τιμές
color, style, width = flow_styles['SOC']
ax2.plot(daily_stochastic["Timestep_in_Day"], daily_stochastic["SOC"],
         label='SOC (%)', color=color, linestyle=style, linewidth=width, alpha=0.9)

color, style, width = flow_styles['Buy Price']
ax2.plot(daily_stochastic["Timestep_in_Day"], daily_stochastic["BuyPrice"],
         label='Τιμή Αγοράς (€/kWh)', color=color, linestyle=style, linewidth=width, alpha=0.8)

color, style, width = flow_styles['Sell Price']
ax2.plot(daily_stochastic["Timestep_in_Day"], daily_stochastic["SellPrice"],
         label='Τιμή Πώλησης (€/kWh)', color=color, linestyle=style, linewidth=width, alpha=0.8)

# Ρυθμίσεις αξόνων
ax1.set_ylabel("Ενέργεια (kWh)", fontsize=14, fontweight='bold')
ax1.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
ax1.tick_params(axis='y', labelsize=12)
ax1.tick_params(axis='x', labelsize=11)
ax1.set_ylim(bottom=0)

ax2.set_ylabel("SOC (%) / Τιμή (€/kWh)", fontsize=14, fontweight='bold')
ax2.tick_params(axis='y', labelsize=12)
ax2.set_ylim(bottom=0)

# Ρύθμιση X axis - εμφάνιση timesteps
ax1.set_xlabel("Timestep της Ημέρας", fontsize=14, fontweight='bold')
ax1.set_xticks(daily_stochastic["Timestep_in_Day"])

# Legend με καλύτερη οργάνωση
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='upper left', fontsize=11, framealpha=0.95, ncol=3,
           bbox_to_anchor=(0, 1), borderaxespad=0)

ax1.set_title(f"Stochastic Optimization - Ημέρα {SELECTED_DAY}",
              fontsize=15, fontweight='bold', pad=15)

# ===== SUBPLOT 2: MILP =====
ax4 = ax3.twinx()

# Ροές ενέργειας
color, style, width = flow_styles['PV→Demand']
ax3.plot(daily_milp["Timestep_in_Day"], daily_milp["Ppv_d"],
         label='PV → Ζήτηση', color=color, linestyle=style, linewidth=width, alpha=0.85)

color, style, width = flow_styles['PV→Battery']
ax3.plot(daily_milp["Timestep_in_Day"], daily_milp["Ppv_b"],
         label='PV → Μπαταρία', color=color, linestyle=style, linewidth=width, alpha=0.85)

color, style, width = flow_styles['PV→Grid']
ax3.plot(daily_milp["Timestep_in_Day"], daily_milp["Ppv_g"],
         label='PV → Δίκτυο', color=color, linestyle=style, linewidth=width, alpha=0.85)

color, style, width = flow_styles['Grid Import']
ax3.plot(daily_milp["Timestep_in_Day"], daily_milp["Pg_import"],
         label='Αγορά από Δίκτυο', color=color, linestyle=style, linewidth=width, alpha=0.85)

color, style, width = flow_styles['Grid Export']
ax3.plot(daily_milp["Timestep_in_Day"], daily_milp["Pg_export"],
         label='Πώληση σε Δίκτυο', color=color, linestyle=style, linewidth=width, alpha=0.85)

color, style, width = flow_styles['Battery Charge']
ax3.plot(daily_milp["Timestep_in_Day"], daily_milp["Pb_c"],
         label='Φόρτιση Μπαταρίας', color=color, linestyle=style, linewidth=width, alpha=0.85)

color, style, width = flow_styles['Battery Discharge']
ax3.plot(daily_milp["Timestep_in_Day"], daily_milp["Pdisc"],
         label='Εκφόρτιση Μπαταρίας', color=color, linestyle=style, linewidth=width, alpha=0.85)

# SOC και τιμές
color, style, width = flow_styles['SOC']
ax4.plot(daily_milp["Timestep_in_Day"], daily_milp["SOC"],
         label='SOC (%)', color=color, linestyle=style, linewidth=width, alpha=0.9)

color, style, width = flow_styles['Buy Price']
ax4.plot(daily_milp["Timestep_in_Day"], daily_milp["BuyPrice"],
         label='Τιμή Αγοράς (€/kWh)', color=color, linestyle=style, linewidth=width, alpha=0.8)

color, style, width = flow_styles['Sell Price']
ax4.plot(daily_milp["Timestep_in_Day"], daily_milp["SellPrice"],
         label='Τιμή Πώλησης (€/kWh)', color=color, linestyle=style, linewidth=width, alpha=0.8)

# Ρυθμίσεις αξόνων
ax3.set_xlabel("Timestep της Ημέρας", fontsize=14, fontweight='bold')
ax3.set_ylabel("Ενέργεια (kWh)", fontsize=14, fontweight='bold')
ax3.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
ax3.tick_params(axis='y', labelsize=12)
ax3.tick_params(axis='x', labelsize=11)
ax3.set_ylim(bottom=0)

ax4.set_ylabel("SOC (%) / Τιμή (€/kWh)", fontsize=14, fontweight='bold')
ax4.tick_params(axis='y', labelsize=12)
ax4.set_ylim(bottom=0)

# Ρύθμιση X axis
ax3.set_xticks(daily_milp["Timestep_in_Day"])

# Legend
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4,
           loc='upper left', fontsize=11, framealpha=0.95, ncol=3,
           bbox_to_anchor=(0, 1), borderaxespad=0)

ax3.set_title(f"MILP Deterministic Optimization - Ημέρα {SELECTED_DAY}",
              fontsize=15, fontweight='bold', pad=15)

# ===== ΚΕΝΤΡΙΚΟΣ ΤΙΤΛΟΣ =====
fig.suptitle(f"Σύγκριση Ροών Ενέργειας: Stochastic vs MILP\nΗμέρα {SELECTED_DAY} - Ανά Timestep",
             fontsize=17, fontweight='bold', y=0.998)

plt.tight_layout(rect=[0, 0, 1, 0.985])

# Αποθήκευση
filename = f'συγκριση_Stochastic_vs_MILP_ημερα_{SELECTED_DAY}.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Διάγραμμα για ημέρα {SELECTED_DAY} δημιουργήθηκε: {filename}")
print(f"   - Timesteps: {len(daily_stochastic)}")
print("\n🎉 Ολοκληρώθηκε!")
