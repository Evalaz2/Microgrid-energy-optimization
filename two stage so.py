import pandas as pd
from pyscipopt import Model
import numpy as np
import matplotlib.pyplot as plt

# ===== ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ =====

# ΣΗΜΑΝΤΙΚΟ: Χρησιμοποίησε το ΔΙΟΡΘΩΜΕΝΟ αρχείο σεναρίων
df = pd.read_excel(r"C:\Users\30697\Desktop\ΔΙΠΛΩΜΑΤΙΚΗ\monte_carlo_scenarios_CORRECTED.xlsx")

num_scenarios = df['scenario'].nunique()
DAY_STEPS = 96

# Δημιουργία στήλης Day αν δεν υπάρχει
if 'Day' not in df.columns:
    df['Day'] = (df['timestep'] // DAY_STEPS).astype(int)  # 0-based days

# Φόρτωση first-stage αποτελεσμάτων
first_stage = pd.read_excel('first_stage_2.xlsx')
num_days = int(first_stage['Day'].max())

print("="*70)
print("ΔΕΔΟΜΕΝΑ ΦΟΡΤΩΘΗΚΑΝ")
print("="*70)
print(f"Σενάρια: {num_scenarios}")
print(f"Ημέρες: {num_days}")
print(f"Timesteps/day: {DAY_STEPS}")
print(f"Συνολικά timesteps: {len(df) // num_scenarios}")

# ===== ΠΙΘΑΝΟΤΗΤΕΣ ΣΕΝΑΡΙΩΝ =====

# Χρήση ομοιόμορφων πιθανοτήτων (equal likelihood)
scenario_probs = np.ones(num_scenarios) / num_scenarios

print("\nΠιθανότητες Σεναρίων (Ομοιόμορφες):")
for i, p in enumerate(scenario_probs):
    print(f"  Scenario {i}: {p:.6f}")
print(f"Άθροισμα: {scenario_probs.sum():.10f}\n")

# ===== ΠΑΡΑΜΕΤΡΟΙ ΣΥΣΤΗΜΑΤΟΣ =====

delta_t = 0.25
SOC_max = 1
SOC_min = 0
eta_c = 0.98
eta_d = 0.95
Ppv_max = 3.5
delta_c = 5  
delta_esb = delta_c/2
Pdiscmax = delta_c
Pb_cmax = delta_c
daily_costs = []
all_results = []

print("="*70)
print("ΕΝΑΡΞΗ ΒΕΛΤΙΣΤΟΠΟΙΗΣΗΣ")
print("="*70)

# === FIRST-STAGE METAΒΛΗΤΕΣ (ΚΟΙΝΕΣ ΓΙΑ ΟΛΑ ΤΑ ΣΕΝΑΡΙΑ) ===
T = len(first_stage)
day_steps = 96  # 15 λεπτά ανά βήμα, άρα 96 βήματα/μέρα
num_days = T // day_steps


for day in range(num_days):
    f = first_stage[first_stage['Day'] == (day+1)].reset_index(drop=True)
    timesteps = len(f)
    start_t = day * day_steps
    end_t = (day + 1) * day_steps
    print(f"\n=== ΜΕΡΑ {day+1} ===")
    # δημιουργία μοντέλου για κάθε μέρα
    model = Model("StochasticEnergyMILP")
    model.setParam('limits/time', 300) 
    # Επιλογή δεδομένων της ημέρας
    Pb_d1   = f["Pb_d"].values
    Ppv_d1  = f["Ppv_d"].values
    Ppv_b1  = f["Ppv_b"].values
    Ppv_g1  = f["Ppv_g"].values
    Pb_g1   = f["Pb_g"].values
    Pg_d1   = f["Pg_d"].values
    Pg_b1   = f["Pg_b"].values
    Pb_c1   = f["Pb_c"].values
    Pdisc1  = f["Pdisc"].values
    Pg_import1 = f["Pg_import"].values
    Pg_export1 = f["Pg_export"].values
    BuyPrice1 = f["BuyPrice"].values
    SellPrice1 = f["SellPrice"].values

    
    
    # === SECOND-STAGE METAΒΛΗΤΕΣ (ΓΙΑ ΚΑΘΕ ΣΕΝΑΡΙΟ ΚΑΙ TIMESTEP) ===
    vars_scenarios = {}
    
    for s in range(num_scenarios):
        vars_s = {}
        for t in range(timesteps):
            vars_s[f"Ppv_b_2_{t}"] = model.addVar(lb=0, ub=Ppv_max, name=f"Ppv_b_{t}_{s}")
            vars_s[f"Ppv_g_2_{t}"] = model.addVar(lb=0, ub=Ppv_max, name=f"Ppv_g_{t}_{s}")
            vars_s[f"Ppv_d_2_{t}"] = model.addVar(lb=0, ub=Ppv_max, name=f"Ppv_d_{t}_{s}")
            vars_s[f"Pdisc_2_{t}"] = model.addVar(lb=0,  name=f"Pdisc_{t}_{s}")
            vars_s[f"Pb_d_2_{t}"] = model.addVar(lb=0, name=f"Pb_d_{t}_{s}")
            vars_s[f"Pb_g_2_{t}"] = model.addVar(lb=0,  name=f"Pb_g_{t}_{s}")
            vars_s[f"Pg_d_2_{t}"] = model.addVar(lb=0, name=f"Pg_d_{t}_{s}")
            vars_s[f"Pg_b_2_{t}"] = model.addVar(lb=0, name=f"Pg_b_{t}_{s}")
            vars_s[f"yb_c_2_{t}"] = model.addVar(vtype="B", name=f"yb_c_{t}_{s}")
            vars_s[f"yb_d_2_{t}"] = model.addVar(vtype="B", name=f"yb_d_{t}_{s}")
            vars_s[f"Pb_c_2_{t}"] = model.addVar(lb=0,  name=f"Pb_c_{t}_{s}")
            vars_s[f"SOC_2_{t}"] = model.addVar(lb=SOC_min, ub=SOC_max, name=f"SOC_{t}_{s}")
            vars_s[f"Pg_import_2_{t}"] = model.addVar(lb=0, name=f"Pg_import_{t}_{s}")
            vars_s[f"Pg_export_2_{t}"] = model.addVar(lb=0, name=f"Pg_export_{t}_{s}")
        vars_scenarios[s] = vars_s
        # Φιλτράρισμα
    df_s = df[(df['scenario'] == s) & 
              (df['timestep'] >= start_t) & 
              (df['timestep'] < end_t)].reset_index(drop=True)
    

    # === ΠΕΡΙΟΡΙΣΜΟΙ ===
    for s in range(num_scenarios):
        print(f" Σενάριο {s}...")
        vars_s = vars_scenarios[s]
        df_s = df[(df['scenario'] == s) & 
              (df['timestep'] >= start_t) & 
              (df['timestep'] < end_t)].reset_index(drop=True)
    
        
        for t in range(timesteps):
            local_t=t-start_t
            load1=f.loc[t,"Load"]
            prod1= f.loc[t,"Production"]
            load2 = df_s.loc[t, "Load"]
            prod2 = df_s.loc[t, "PV"]

            # 3.10 Ισοζύγιο φορτίου
            model.addCons(
                (load1 + load2) == (
                    (Ppv_d1[t] + vars_s[f"Ppv_d_2_{t}"])
                    + (Pb_d1[t] + vars_s[f"Pb_d_2_{t}"])
                    + (Pg_d1[t] + vars_s[f"Pg_d_2_{t}"])
                )
            )

            # 3.11 Μη ταυτόχρονη φόρτιση/εκφόρτιση μπαταριας
            model.addCons(vars_s[f"yb_c_2_{t}"] + vars_s[f"yb_d_2_{t}"]<=1)

            # 3./12 Εκφόρτιση μόνο όταν yb_d=1

            model.addCons((Pdisc1[t] + vars_s[f"Pdisc_2_{t}"]) <= Pdiscmax * vars_s[f"yb_d_2_{t}"])

            # 3.13 Φόρτιση μόνο όταν yb_c=1

            model.addCons((Pb_c1[t] + vars_s[f"Pb_c_2_{t}"]) <= Pb_cmax *  vars_s[f"yb_c_2_{t}"])

            # 3.14 Ενέργεια φόρτισης ESS

            model.addCons((Pb_c1[t] + vars_s[f"Pb_c_2_{t}"]) == (Pg_b1[t] + vars_s[f"Pg_b_2_{t}"]) + (Ppv_b1[t] + vars_s[f"Ppv_b_2_{t}"]))

            # 3.15. Περιορισμός παραγωγής PV

            model.addCons(prod1 + prod2 == (Ppv_d1[t] + vars_s[f"Ppv_d_2_{t}"]) + (Ppv_b1[t] + vars_s[f"Ppv_b_2_{t}"]) + (Ppv_g1[t] + vars_s[f"Ppv_g_2_{t}"]))

            # 3.16 Εισαγωγή από δίκτυο

            model.addCons((Pg_import1[t] + vars_s[f"Pg_import_2_{t}"]) == (Pg_d1[t] + vars_s[f"Pg_d_2_{t}"]) + (Pg_b1[t] + vars_s[f"Pg_b_2_{t}"]))

            # 3.17 Εξαγωγή στο δίκτυο

            model.addCons((Pg_export1[t] + vars_s[f"Pg_export_2_{t}"]) == (Ppv_g1[t] + vars_s[f"Ppv_g_2_{t}"]) + (Pb_g1[t] + vars_s[f"Pb_g_2_{t}"]))

            # 3.18 Εκφόρτιση μπαταρίας

            model.addCons((Pdisc1[t] + vars_s[f"Pdisc_2_{t}"]) == (Pb_d1[t] + vars_s[f"Pb_d_2_{t}"]) + (Pb_g1[t] + vars_s[f"Pb_g_2_{t}"]))

            #Περιορισμοί μη αρνητικότητας

            model.addCons(Ppv_g1[t] + vars_s[f"Ppv_g_2_{t}"] >= 0)
            model.addCons(Ppv_d1[t] + vars_s[f"Ppv_d_2_{t}"] >= 0)
            model.addCons(Ppv_b1[t] + vars_s[f"Ppv_b_2_{t}"] >= 0)
            model.addCons(Pb_d1[t] + vars_s[f"Pb_d_2_{t}"] >= 0)
            model.addCons(Pb_g1[t] + vars_s[f"Pb_g_2_{t}"] >= 0)
            model.addCons(Pg_d1[t] + vars_s[f"Pg_d_2_{t}"] >= 0)
            model.addCons(Pg_b1[t] + vars_s[f"Pg_b_2_{t}"] >= 0)

            # 12. SOC limits

            for t in range(timesteps):
                if t==0:
                   model.addCons(vars_s[f"SOC_2_{t}"] * delta_c == delta_esb) #3.22
                else:
                    model.addCons(vars_s[f"SOC_2_{t}"] == vars_s[f"SOC_2_{t-1}"]+ ( ( (Pb_c1[t] + vars_s[f"Pb_c_2_{t}"])*eta_c /  delta_c ) - ( (Pdisc1[t] + vars_s[f"Pdisc_2_{t}"]) /(eta_d* delta_c) )  ) * delta_t)  #3.21
                   
            
    # OBJECTIVE: Μέσο κόστος σεναρίων 
    objective_terms = []
    for t in range(timesteps):
        buy1=f.loc[t,"BuyPrice"]
        sell1=f.loc[t,"SellPrice"]
        objective_terms.append(Pg_import1[t]* buy1 *delta_t - Pg_export1[t]* sell1 *delta_t)
    
    for s in range(num_scenarios):
        
        vars_s = vars_scenarios[s]
        prob = scenario_probs[s]
        for t in range(timesteps):
            buy2 = df_s.loc[t, "BuyPrice"]
            sell2 = df_s.loc[t, "SellPrice"]
            objective_terms.append( scenario_probs[s] *( vars_s[f"Pg_import_2_{t}"] * buy2 *delta_t - vars_s[f"Pg_export_2_{t}"] * sell2 *delta_t))
    model.setObjective(sum(objective_terms), "minimize")

    #  Επίλυση 
    model.optimize()

    if model.getStatus() == "optimal":
        first_stage_cost = 0.0

        for t in range(timesteps):
            buy1 = f.loc[t, "BuyPrice"]
            sell1 = f.loc[t, "SellPrice"]
            first_stage_cost += (Pg_import1[t] * buy1 - Pg_export1[t] * sell1) * delta_t
        
        # Συλλογή αποτελεσμάτων για ΚΑΘΕ σενάριο
        scenario_costs = []
        scenario_results = {}  # Λεξικό: scenario -> list of timestep results
        
        for s in range(num_scenarios):
            vars_s = vars_scenarios[s]
            df_s = df[(df['scenario'] == s) &
                    (df['timestep'] >= start_t) &
                    (df['timestep'] < end_t)].reset_index(drop=True)
            
            # Υπολογισμός κόστους σεναρίου
            scen_cost_2 = 0
            for t in range(timesteps):
                buy2 = df_s.loc[t, "BuyPrice"]
                sell2 = df_s.loc[t, "SellPrice"]
                scen_cost_2 += model.getVal(vars_s[f"Pg_import_2_{t}"]) * buy2 * delta_t \
                            - model.getVal(vars_s[f"Pg_export_2_{t}"]) * sell2 * delta_t
            
            total_cost = first_stage_cost + scen_cost_2
            scenario_costs.append(total_cost)
            print(f"Μέρα {day+1}, Σενάριο {s}: Κόστος = {total_cost:.2f} €")
            
            # Συλλογή ροών για κάθε timestep
            scenario_results[s] = []
            for t in range(timesteps):
                load1 = f.loc[t, "Load"]
                prod1 = f.loc[t, "Production"]
                load2 = df_s.loc[t, "Load"]
                prod2 = df_s.loc[t, "PV"]
                
                # Υπολογισμός συνολικών ροών (Stage 1 + Stage 2)
                result_t = {
                    'total_Ppv_d': Ppv_d1[t] + model.getVal(vars_s[f"Ppv_d_2_{t}"]),
                    'total_Ppv_b': Ppv_b1[t] + model.getVal(vars_s[f"Ppv_b_2_{t}"]),
                    'total_Ppv_g': Ppv_g1[t] + model.getVal(vars_s[f"Ppv_g_2_{t}"]),
                    'total_Pb_d': Pb_d1[t] + model.getVal(vars_s[f"Pb_d_2_{t}"]),
                    'total_Pb_g': Pb_g1[t] + model.getVal(vars_s[f"Pb_g_2_{t}"]),
                    'total_Pb_c': Pb_c1[t] + model.getVal(vars_s[f"Pb_c_2_{t}"]),
                    'total_Pg_d': Pg_d1[t] + model.getVal(vars_s[f"Pg_d_2_{t}"]),
                    'total_Pg_b': Pg_b1[t] + model.getVal(vars_s[f"Pg_b_2_{t}"]),
                    'total_Pdisc': Pdisc1[t] + model.getVal(vars_s[f"Pdisc_2_{t}"]),
                    'total_Pg_import': Pg_import1[t] + model.getVal(vars_s[f"Pg_import_2_{t}"]),
                    'total_Pg_export': Pg_export1[t] + model.getVal(vars_s[f"Pg_export_2_{t}"]),
                    'total_SOC': model.getVal(vars_s[f"SOC_2_{t}"]),
                    'load': load1 + load2,
                    'pv_production': prod1 + prod2,
                    'buy_price': df_s.loc[t, "BuyPrice"],
                    'sell_price': df_s.loc[t, "SellPrice"]
                }
                scenario_results[s].append(result_t)
        
        # Υπολογισμός μέσου κόστους ημέρας
        avg_day_cost = np.average(scenario_costs, weights=scenario_probs)
        print(f"Μέσο κόστος ημέρας {day+1}: {avg_day_cost:.2f} €")
        daily_costs.append(float(avg_day_cost))
        
        # Υπολογισμός αναμενόμενων τιμών ροών ανά timestep
        avg_flows = []
        for t in range(timesteps):
            # Συλλογή τιμών για timestep t από όλα τα σενάρια
            flows_t = {
                'avg_Ppv_d': 0, 'avg_Ppv_b': 0, 'avg_Ppv_g': 0,
                'avg_Pb_d': 0, 'avg_Pb_g': 0, 'avg_Pb_c': 0,
                'avg_Pg_d': 0, 'avg_Pg_b': 0, 'avg_Pdisc': 0,
                'avg_Pg_import': 0, 'avg_Pg_export': 0, 'avg_SOC': 0
            }
            
            for s in range(num_scenarios):
                result_t = scenario_results[s][t]
                prob = scenario_probs[s]
                flows_t['avg_Ppv_d'] += prob * result_t['total_Ppv_d']
                flows_t['avg_Ppv_b'] += prob * result_t['total_Ppv_b']
                flows_t['avg_Ppv_g'] += prob * result_t['total_Ppv_g']
                flows_t['avg_Pb_d'] += prob * result_t['total_Pb_d']
                flows_t['avg_Pb_g'] += prob * result_t['total_Pb_g']
                flows_t['avg_Pb_c'] += prob * result_t['total_Pb_c']
                flows_t['avg_Pg_d'] += prob * result_t['total_Pg_d']
                flows_t['avg_Pg_b'] += prob * result_t['total_Pg_b']
                flows_t['avg_Pdisc'] += prob * result_t['total_Pdisc']
                flows_t['avg_Pg_import'] += prob * result_t['total_Pg_import']
                flows_t['avg_Pg_export'] += prob * result_t['total_Pg_export']
                flows_t['avg_SOC'] += prob * result_t['total_SOC']
            
            avg_flows.append(flows_t)
        
        # Αποθήκευση στο all_results
        for s in range(num_scenarios):
            for t in range(timesteps):
                global_t = start_t + t
                result_t = scenario_results[s][t]
                flows_t = avg_flows[t]
                
                all_results.append({
                    'Day': day + 1,
                    'Scenario': s,
                    'Scenario_Probability': scenario_probs[s], 
                    'Timestep_in_Day': t,
                    'Global_Timestep': global_t,
                    'Load': result_t['load'],
                    'PV_Production': result_t['pv_production'],
                    'BuyPrice': result_t['buy_price'],
                    'SellPrice': result_t['sell_price'],
                    # Συνολικές ροές για αυτό το σενάριο
                    'Total_Ppv_d': result_t['total_Ppv_d'],
                    'Total_Ppv_b': result_t['total_Ppv_b'],
                    'Total_Ppv_g': result_t['total_Ppv_g'],
                    'Total_Pb_d': result_t['total_Pb_d'],
                    'Total_Pb_g': result_t['total_Pb_g'],
                    'Total_Pb_c': result_t['total_Pb_c'],
                    'Total_Pg_d': result_t['total_Pg_d'],
                    'Total_Pg_b': result_t['total_Pg_b'],
                    'Total_Pdisc': result_t['total_Pdisc'],
                    'Total_Pg_import': result_t['total_Pg_import'],
                    'Total_Pg_export': result_t['total_Pg_export'],
                    'SOC': result_t['total_SOC'],
                    # Αναμενόμενες ροές (scalar values, όχι arrays!)
                    'avg_Ppv_d': flows_t['avg_Ppv_d'],
                    'avg_Ppv_b': flows_t['avg_Ppv_b'],
                    'avg_Ppv_g': flows_t['avg_Ppv_g'],
                    'avg_Pb_d': flows_t['avg_Pb_d'],
                    'avg_Pb_g': flows_t['avg_Pb_g'],
                    'avg_Pb_c': flows_t['avg_Pb_c'],
                    'avg_Pg_d': flows_t['avg_Pg_d'],
                    'avg_Pg_b': flows_t['avg_Pg_b'],
                    'avg_Pdisc': flows_t['avg_Pdisc'],
                    'avg_Pg_import': flows_t['avg_Pg_import'],
                    'avg_Pg_export': flows_t['avg_Pg_export'],
                    'avg_SOC': flows_t['avg_SOC']
                })
        
    else:
        daily_costs.append(None)
        print(f"Δεν βρέθηκε βέλτιστη λύση για τη μέρα {day+1}.")
    daily_costs.append(float(avg_day_cost))
        

results_df = pd.DataFrame(all_results)
results_df.to_excel('stochastic_optimization_results.xlsx', index=False, engine='openpyxl')
print(f"\nΤα αποτελέσματα αποθηκεύτηκαν στο: stochastic_optimization_results.xlsx ")
print(f"Συνολικές γραμμές: {len(results_df)}")
