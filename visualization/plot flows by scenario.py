import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# ═══════════════════════════════════════════════════════════════════════
# ΡΥΘΜΙΣΕΙΣ
# ═══════════════════════════════════════════════════════════════════════

EXCEL_FILE = 'stochastic_optimization_results.xlsx'
DAY_TO_PLOT = 1           
NUM_SCENARIOS = 4            # Πόσα σενάρια να δείξει
FIGURE_SIZE = (18, 7)
DPI = 300

# ═══════════════════════════════════════════════════════════════════════


def load_data(excel_file):
    """Φόρτωση δεδομένων από Excel"""
    print("="*70)
    print("ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ")
    print("="*70)
    
    if not os.path.exists(excel_file):
        print(f"\n❌ ΣΦΑΛΜΑ: Δεν βρέθηκε το αρχείο: {excel_file}\n")
        return None
    
    try:
        df = pd.read_excel(excel_file, engine='openpyxl')
        print(f"✓ Φορτώθηκε: {excel_file}")
        print(f"✓ Γραμμές: {len(df)}")
        print(f"✓ Ημέρες: {sorted(df['Day'].unique())}")
        print(f"✓ Σενάρια: {sorted(df['Scenario'].unique())}")
        return df
    except Exception as e:
        print(f"\n❌ ΣΦΑΛΜΑ: {e}\n")
        return None

def get_scenario_probabilities_from_excel(df):
    """
    Διαβάζει τις πιθανότητες σεναρίων από το Excel
    (από τη στήλη Scenario_Probability)
    """
    print(f"\nΛήψη πιθανοτήτων σεναρίων από το Excel...")
    
    if 'Scenario_Probability' not in df.columns:
        print("⚠️ ΠΡΟΣΟΧΗ: Δεν βρέθηκε η στήλη 'Scenario_Probability'!")
        print("Χρήση ίσων πιθανοτήτων για όλα τα σενάρια.")
        num_scenarios = df['Scenario'].nunique()
        return np.ones(num_scenarios) / num_scenarios
    
    # Παίρνουμε τις πιθανότητες
    scenario_probs = df.groupby('Scenario')['Scenario_Probability'].first().values
    
    print(f"✓ Βρέθηκαν πιθανότητες από το Excel:")
    for i, p in enumerate(scenario_probs):
        print(f"   Σενάριο {i}: {p:.6f} ({p*100:.2f}%)")
    
    print(f"\nΆθροισμα πιθανοτήτων: {scenario_probs.sum():.10f}")
    
    if abs(scenario_probs.sum() - 1.0) > 1e-6:
        print("⚠️ ΠΡΟΣΟΧΗ: Οι πιθανότητες ΔΕΝ αθροίζουν σε 1!")
    else:
        print("✓ Οι πιθανότητες αθροίζουν σωστά σε 1")
    
    return scenario_probs

def plot_single_scenario(df, day, scenario, save_fig=True):
    """Διάγραμμα για ένα συγκεκριμένο σενάριο"""
    
    print(f"\n{'='*70}")
    print(f"ΣΕΝΑΡΙΟ {scenario} - Ημέρα {day}")
    print("="*70)
    
    data = df[(df['Day'] == day) & (df['Scenario'] == scenario)].copy()
    data = data.sort_values('Timestep_in_Day').reset_index(drop=True)
    
    if len(data) == 0:
        print(f"⚠️  Δεν υπάρχουν δεδομένα")
        return None
    
    timesteps = data['Timestep_in_Day'].values
    
    fig, ax1 = plt.subplots(figsize=FIGURE_SIZE)
    ax2 = ax1.twinx()
    
    # PV ροές
    ax1.plot(timesteps, data['Total_Ppv_d'].values, '-', linewidth=2.5, 
             label='PV → Load', color='#FFD700', alpha=0.9)
    ax1.plot(timesteps, data['Total_Ppv_b'].values, '--', linewidth=1.8, 
             label='PV → Battery', color='#FFA500', alpha=0.8)
    ax1.plot(timesteps, data['Total_Ppv_g'].values, ':', linewidth=2.2, 
             label='PV → Grid', color='#FF8C00', alpha=0.8)
    
    # Battery ροές
    ax1.plot(timesteps, data['Total_Pb_d'].values, '-', linewidth=2.5, 
             label='Battery → Load', color='#32CD32', alpha=0.9)
    ax1.plot(timesteps, data['Total_Pb_g'].values, '--', linewidth=1.8, 
             label='Battery → Grid', color='#228B22', alpha=0.8)
    ax1.plot(timesteps, data['Total_Pb_c'].values, '-', linewidth=1.6, 
             label='Battery Charge', color='#00CED1', alpha=0.7)
    ax1.plot(timesteps, data['Total_Pdisc'].values, '-', linewidth=1.6, 
             label='Battery Discharge', color='#4682B4', alpha=0.7)
    
    # Grid ροές
    ax1.plot(timesteps, data['Total_Pg_d'].values, '-.', linewidth=2.2, 
             label='Grid → Load', color='#4169E1', alpha=0.9)
    ax1.plot(timesteps, data['Total_Pg_b'].values, ':', linewidth=1.8, 
             label='Grid → Battery', color='#6495ED', alpha=0.8)
    ax1.plot(timesteps, data['Total_Pg_import'].values, '-', linewidth=3.5, 
             label='Grid Import', color='#000080', alpha=1.0)
    ax1.plot(timesteps, data['Total_Pg_export'].values, '-', linewidth=2.5, 
             label='Grid Export', color='#87CEEB', alpha=0.8)
    
    # Τιμές
    ax2.plot(timesteps, data['BuyPrice'].values, '--', linewidth=2.5, 
             label='Τιμή Αγοράς (€/kWh)', color='#000000', alpha=0.9)
    ax2.plot(timesteps, data['SellPrice'].values, ':', linewidth=2.5, 
             label='Τιμή Πώλησης (€/kWh)', color='#FF00FF', alpha=0.9)
    
    # Μορφοποίηση
    ax1.set_title(f'Ροές Ενέργειας - Ημέρα {day+1}, Σενάριο {scenario}', 
                  fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Χρονικό βήμα (15λεπτα)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Ενέργεια (kWh)', fontsize=13, fontweight='bold', color='blue')
    ax2.set_ylabel('Τιμή (€/kWh)', fontsize=13, fontweight='bold', color='purple')
    
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='purple')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper left', bbox_to_anchor=(1.12, 1.0),
              fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    
    if save_fig:
        filename = f'scenario_{scenario}_day{day}.png'
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        print(f"✓ Αποθηκεύτηκε: {filename}")
    
    plt.show()
    return fig


def plot_weighted_average(df, scenario_probs, day, save_fig=True):
    """Διάγραμμα με ΣΤΑΘΜΙΣΜΕΝΟ μέσο όρο (πιθανότητες από TWO-STAGE-SO-2)"""
    
    print(f"\n{'='*70}")
    print(f"ΣΤΑΘΜΙΣΜΕΝΟΣ ΜΕΣΟΣ ΟΡΟΣ (ΜΕ ΠΙΘΑΝΟΤΗΤΕΣ) - Ημέρα {day}")
    print("="*70)
    
    # Φιλτράρισμα για την ημέρα
    day_data = df[df['Day'] == day].copy()
    
    if len(day_data) == 0:
        print(f"⚠️  Δεν υπάρχουν δεδομένα για Ημέρα {day}")
        return None
    
    # Προσθήκη πιθανοτήτων
    day_data['probability'] = day_data['Scenario'].map(
        lambda s: scenario_probs[s] if s < len(scenario_probs) else 0
    )
    
    # ΣΤΑΘΜΙΣΜΕΝΟΣ μέσος όρος ανά timestep
    weighted_avg = day_data.groupby('Timestep_in_Day').apply(
        lambda x: pd.Series({
            'Total_Ppv_d': (x['Total_Ppv_d'] * x['probability']).sum(),
            'Total_Ppv_b': (x['Total_Ppv_b'] * x['probability']).sum(),
            'Total_Ppv_g': (x['Total_Ppv_g'] * x['probability']).sum(),
            'Total_Pb_d': (x['Total_Pb_d'] * x['probability']).sum(),
            'Total_Pb_g': (x['Total_Pb_g'] * x['probability']).sum(),
            'Total_Pb_c': (x['Total_Pb_c'] * x['probability']).sum(),
            'Total_Pdisc': (x['Total_Pdisc'] * x['probability']).sum(),
            'Total_Pg_d': (x['Total_Pg_d'] * x['probability']).sum(),
            'Total_Pg_b': (x['Total_Pg_b'] * x['probability']).sum(),
            'Total_Pg_import': (x['Total_Pg_import'] * x['probability']).sum(),
            'Total_Pg_export': (x['Total_Pg_export'] * x['probability']).sum(),
            'BuyPrice': (x['BuyPrice'] * x['probability']).sum(),
            'SellPrice': (x['SellPrice'] * x['probability']).sum()
        })
    ).reset_index()
    
    timesteps = weighted_avg['Timestep_in_Day'].values
    
    print(f"✓ Υπολογίστηκε σταθμισμένος μέσος όρος με τις ίδιες πιθανότητες του TWO-STAGE-SO-2")
    
    # Δημιουργία διαγράμματος
    fig, ax1 = plt.subplots(figsize=FIGURE_SIZE)
    ax2 = ax1.twinx()
    
    # PV ροές
    ax1.plot(timesteps, weighted_avg['Total_Ppv_d'].values, '-', linewidth=2.5, 
             label='PV → Load', color='#FFD700', alpha=0.9)
    ax1.plot(timesteps, weighted_avg['Total_Ppv_b'].values, '--', linewidth=1.8, 
             label='PV → Battery', color='#FFA500', alpha=0.8)
    ax1.plot(timesteps, weighted_avg['Total_Ppv_g'].values, ':', linewidth=2.2, 
             label='PV → Grid', color='#FF8C00', alpha=0.8)
    
    # Battery ροές
    ax1.plot(timesteps, weighted_avg['Total_Pb_d'].values, '-', linewidth=2.5, 
             label='Battery → Load', color='#32CD32', alpha=0.9)
    ax1.plot(timesteps, weighted_avg['Total_Pb_g'].values, '--', linewidth=1.8, 
             label='Battery → Grid', color='#228B22', alpha=0.8)
    ax1.plot(timesteps, weighted_avg['Total_Pb_c'].values, '-', linewidth=1.6, 
             label='Battery Charge', color='#00CED1', alpha=0.7)
    ax1.plot(timesteps, weighted_avg['Total_Pdisc'].values, '-', linewidth=1.6, 
             label='Battery Discharge', color='#4682B4', alpha=0.7)
    
    # Grid ροές
    ax1.plot(timesteps, weighted_avg['Total_Pg_d'].values, '-.', linewidth=2.2, 
             label='Grid → Load', color='#4169E1', alpha=0.9)
    ax1.plot(timesteps, weighted_avg['Total_Pg_b'].values, ':', linewidth=1.8, 
             label='Grid → Battery', color='#6495ED', alpha=0.8)
    ax1.plot(timesteps, weighted_avg['Total_Pg_import'].values, '-', linewidth=3.5, 
             label='Grid Import', color='#000080', alpha=1.0)
    ax1.plot(timesteps, weighted_avg['Total_Pg_export'].values, '-', linewidth=2.5, 
             label='Grid Export', color='#87CEEB', alpha=0.8)
    
    # Τιμές
    ax2.plot(timesteps, weighted_avg['BuyPrice'].values, '--', linewidth=2.5, 
             label='Τιμή Αγοράς (€/kWh)', color='#000000', alpha=0.9)
    ax2.plot(timesteps, weighted_avg['SellPrice'].values, ':', linewidth=2.5, 
             label='Τιμή Πώλησης (€/kWh)', color='#FF00FF', alpha=0.9)
    
    # Μορφοποίηση
    ax1.set_title(f'Ροές Ενέργειας - Ημέρα {day+1} (ΣΤΑΘΜΙΣΜΕΝΟΣ ΜΕΣΟΣ ΟΡΟΣ)', 
                  fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Χρονικό βήμα (15λεπτα)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Ενέργεια (kWh)', fontsize=13, fontweight='bold', color='blue')
    ax2.set_ylabel('Τιμή (€/kWh)', fontsize=13, fontweight='bold', color='purple')
    
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='purple')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper left', bbox_to_anchor=(1.12, 1.0),
              fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    
    if save_fig:
        filename = f'weighted_average_day{day}.png'
        plt.savefig(filename, dpi=DPI, bbox_inches='tight')
        print(f"✓ Αποθηκεύτηκε: {filename}")
    
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# ΕΚΤΕΛΕΣΗ
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("ΔΙΑΓΡΑΜΜΑΤΑ ΣΕΝΑΡΙΩΝ ΚΑΙ ΣΤΑΘΜΙΣΜΕΝΟΥ ΜΕΣΟΥ ΟΡΟΥ")
    print("="*70 + "\n")
    
    # Φόρτωση δεδομένων αποτελεσμάτων
    df = load_data(EXCEL_FILE)
    
    if df is None:
        exit(1)
    
    
    scenario_probs = get_scenario_probabilities_from_excel(df)

    # ΜΕΡΟΣ 1: Διαγράμματα για τα 4 πρώτα σενάρια
    print(f"\n{'='*70}")
    print(f"ΔΗΜΙΟΥΡΓΙΑ ΔΙΑΓΡΑΜΜΑΤΩΝ ΓΙΑ {NUM_SCENARIOS} ΣΕΝΑΡΙΑ")
    print("="*70)
    
    for scenario in range(NUM_SCENARIOS):
        plot_single_scenario(df, DAY_TO_PLOT, scenario)
    
    # ΜΕΡΟΣ 2: Διάγραμμα σταθμισμένου μέσου όρου
    print(f"\n{'='*70}")
    print("ΔΗΜΙΟΥΡΓΙΑ ΔΙΑΓΡΑΜΜΑΤΟΣ ΣΤΑΘΜΙΣΜΕΝΟΥ ΜΕΣΟΥ ΟΡΟΥ")
    print("="*70)
    
    plot_weighted_average(df, scenario_probs, DAY_TO_PLOT)
    
    # Περίληψη
    print("\n" + "="*70)
    print("✅ ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ!")
    print("="*70)
    print(f"\nΔημιουργήθηκαν:")
    print(f"  • {NUM_SCENARIOS} διαγράμματα σεναρίων (scenario_0 έως scenario_{NUM_SCENARIOS-1})")
    print(f"  • 1 διάγραμμα σταθμισμένου μέσου όρου (weighted_average)")
    print(f"\n✓ Οι πιθανότητες υπολογίστηκαν ΑΚΡΙΒΩΣ όπως στο TWO-STAGE-SO-2.py")
    print("="*70 + "\n")