from pyscipopt import Model
import pandas as pd
import matplotlib.pyplot as plt

# Διαβάζουμε τα δεδομένα
df = pd.read_csv(r"C:\Users\30697\Desktop\ΔΙΠΛΩΜΑΤΙΚΗ\PublicDataset\SampleProsumer_total.csv", sep=";")

df["BuyPrice"] = df["Tariff_Charges"]
df["SellPrice"] = df["DAM_Values"]
df["Load_kW"] = df["Load"] / 0.25
df["Production_kW"] = df["Production"] / 0.25

T = len(df)
day_steps = 96  # 15 λεπτά ανά βήμα, άρα 96 βήματα/μέρα
num_days = T // day_steps
#battery_capacities = [5,50,100,400,1000,1500,2000]
average_daily_costs = []
#for delta_c in battery_capacities:
total_daily_costs=[]
all_days = []   
for day in range(num_days):
    start_t = day * day_steps
    end_t = (day + 1) * day_steps
    model = Model("EnergyMILP")

    #Παραμετροι

    SOC_max = 1   #το μεγιστο ποσοστο χωρητικοτητας της μπαταριας                                   
    SOC_min = 0   #το ελαχιστο ποσοστο χωρητικοτητας της μπαταριας
    delta_c = 5   #χωρητικοτητα της μπαταριας
    delta_esb = delta_c/2  #ενεργεια που ειναι αποθηκευμενη στη μπαταρια αρχικα
    eta_c = 0.98           #αποδοτικοτητα φορτισης της μπαταριας
    eta_d = 0.93            #αποδοτικοτηα εκφορτισης της μπαταριας
    Pdiscmax = delta_c 
    Pb_cmax = delta_c 
    delta_t = 0.25
    Ppv_max=3.5


    #Μεταβλητες
    Ppv_b , Ppv_g , Ppv_d= {}, {}, {} 
    Pb_d, Pb_g, Pdisc = {}, {} ,{}
    Pg_d, Pg_b= {}, {} 
    yb_c, yb_d = {}, {} 
    Pb_c = {}
    SOC = {}
    Pg_import, Pg_export = {}, {} 

    for t in range(start_t,end_t): 
        Ppv_b[t] = model.addVar(lb=0, ub=Ppv_max, name=f"Ppv_b_{t}") #ενεργεια απο pv σε ESS
        Ppv_g[t] = model.addVar(lb=0, ub=Ppv_max, name=f"Ppv_g_{t}") #ενεργεια απο pv στο grid
        Ppv_d[t] = model.addVar(lb=0,ub=Ppv_max,name=f"Ppv_d_{t}")     #ενεργεια απο pv στο φορτιο
        Pdisc[t] = model.addVar(lb=0,ub=Pdiscmax, name=f"Pdisc_{t}")  # ενεργεια που εκφορτιζει η μπαταρια 
        Pb_d[t] = model.addVar(lb=0, ub=Pdiscmax, name=f"Pb_d_{t}") #ενεργεια απο ESS στο φορτιο
        Pb_g[t] = model.addVar(lb=0, ub=Pdiscmax , name=f"Pb_g_{t}") #ενεργεια απο ESS στο grid
        Pg_d[t] = model.addVar(lb=0, name=f"Pg_d_{t}")              #ενεργεια απο grid στο φορτιο
        Pg_b[t] = model.addVar(lb=0 , name=f"Pg_b_{t}")             #ενεργεια απο grid στο ESS
        yb_c[t] = model.addVar(vtype="B", name=f"yb_c_{t}")         #μη ταυτοχρονη φορτιση και εκφορτιση του ESS
        yb_d[t] = model.addVar(vtype="B", name=f"yb_d_{t}")
        Pb_c[t] = model.addVar(lb=0,ub=Pb_cmax, name=f"Pb_c_{t}")              #ενεργεια του ESS για να φορτιστει
        SOC[t] = model.addVar(lb=SOC_min, ub=SOC_max , name=f"SOC_{t}") #ποσοστο φορτισης της μπαταριας
        Pg_import[t] = model.addVar(lb=0, name=f"Pg_import_{t}")        #ενεργεια που παιρνει το φορτιο απο το δικτυο
        Pg_export[t] = model.addVar(lb=0, name=f"Pg_export_{t}")        #ενεργεια που πουλαμε στο δικτυο


    #Περιορισμοι 

    for t in range(start_t,end_t):
        load = df.loc[t, "Load_kW"]
        prod = df.loc[t, "Production_kW"]

        model.addCons(load == Ppv_d[t]+ Pb_d[t] + Pg_d[t])                    #1
        model.addCons(yb_c[t] + yb_d[t] <= 1 )                               #2
        model.addCons(Pdisc[t] <= Pdiscmax * yb_d[t])                        #3
        model.addCons(Pb_c[t] <= Pb_cmax * yb_c[t])                          #4
        model.addCons(Pb_c[t] == Pg_b[t] +Ppv_b[t])                          #5
        model.addCons(prod >= Ppv_d[t] + Ppv_b[t]+ Ppv_g[t])                #6
        model.addCons(Pg_import[t] == Pg_d[t] + Pg_b[t])                     #7
        model.addCons(Pg_export[t] == Ppv_g[t] + Pb_g[t])                    #8 
        model.addCons(Pdisc[t] == Pb_d[t] + Pb_g[t])                         #9
    

    #περιορισμοι για SOC
    for t in range(start_t,end_t):
        if t==start_t:
            model.addCons(SOC[start_t] *delta_c == delta_esb)  
        else:                                                                
            model.addCons(SOC[t] == SOC[t-1]+( Pb_c[t]*eta_c/ delta_c- Pdisc[t] /(eta_d*delta_c))* delta_t )   #13 
                    

    # Αντικειμενική συνάρτηση
    objective = sum(
    Pg_import[t] * df.loc[t, "BuyPrice"] *delta_t  - Pg_export[t] * df.loc[t, "SellPrice"] *delta_t
    for t in range(start_t, end_t)
    )
    model.setObjective(objective, "minimize")

    import time

    t0 = time.perf_counter()


    model.optimize()

    t1 = time.perf_counter()
    print(f"Solve time: {t1-t0:.1f}s")
    print("  Nodes:", model.getNNodes(), "  LP iters:", model.getNLPIterations())
    print("  Status:", model.getStatus())


    if model.getStatus() == "optimal":
        daily_cost = model.getObjVal()
        total_daily_costs.append(daily_cost)
        print(f"Ημέρα {day+1}: Κόστος = {daily_cost:.2f} €")

        daily_data = {
            "timestep": [],
            "Ppv_d": [],
            "Ppv_b": [],
            "Ppv_g": [],
            "Pb_d": [],
            "Pb_g": [],
            "Pg_d": [],
            "Pg_b": [],
            "Pb_c": [],
            "Pdisc": [],
            "SOC": [],
            "Pg_import": [],
            "Pg_export": [],
            "yb_d": [],
            "yb_c": []
        }

        for t in range(start_t,end_t):
            daily_data["timestep"].append(t - start_t)
            daily_data["Ppv_d"].append(model.getVal(Ppv_d[t]))
            daily_data["Ppv_b"].append(model.getVal(Ppv_b[t]))
            daily_data["Ppv_g"].append(model.getVal(Ppv_g[t]))
            daily_data["Pb_d"].append(model.getVal(Pb_d[t]))
            daily_data["Pb_g"].append(model.getVal(Pb_g[t]))
            daily_data["Pg_d"].append(model.getVal(Pg_d[t]))
            daily_data["Pg_b"].append(model.getVal(Pg_b[t]))
            daily_data["Pb_c"].append(model.getVal(Pb_c[t]))
            daily_data["Pdisc"].append(model.getVal(Pdisc[t]))
            daily_data["SOC"].append(model.getVal(SOC[t]))
            daily_data["Pg_import"].append(model.getVal(Pg_import[t]))
            daily_data["Pg_export"].append(model.getVal(Pg_export[t]))
            daily_data["yb_d"].append(model.getVal(yb_d[t]))
            daily_data["yb_c"].append(model.getVal(yb_c[t]))
        
        

        # Μετατροπή σε DataFrame
        day_df = pd.DataFrame(daily_data)
        day_df["Day"] = day + 1  # Προσθήκη αριθμού ημέρας
        day_df["BuyPrice"] = df.loc[start_t:end_t-1, "BuyPrice"].values
        day_df["SellPrice"] = df.loc[start_t:end_t-1, "SellPrice"].values
        day_df["Load"] = df.loc[start_t:end_t-1,"Load_kW"].values
        day_df["Production"] = df.loc[start_t:end_t-1,"Production_kW"].values
        print(day_df.round(2).to_string(index=False))
        all_days.append(day_df)
        

    else:
        total_daily_costs.append(None)
        print(f"Ημέρα {day+1}: Δεν βρέθηκε βέλτιστη λύση.")
            
all_days_df = pd.concat(all_days, ignore_index=True)
all_days_df.to_excel("all_days_flows_soc.xlsx", index=False)

average_cost = sum(total_daily_costs) / len(total_daily_costs)
average_daily_costs.append(average_cost)   
print(f"Μέσο Ημερήσιο Κόστος για {delta_c} : {average_cost:.2f} €")

#bar plot για κοστη ανα μηνα 
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
months = [f"Μήνας {i+1}" for i in range(12)]

monthly_costs = []
start = 0

for days in days_in_month:
    end = start + days
    monthly_sum = sum(total_daily_costs[start:end])  # Εδώ παίρνουμε τις ημέρες που αντιστοιχούν σε κάθε μήνα
    monthly_costs.append(monthly_sum)
    start = end




months = [
  
   "Ιαν", "Φεβ", "Μάρ", "Απρ", "Μάιος", "Ιούν",
    "Ιούλ", "Αυγ", "Σεπ", "Οκτ", "Νοε", "Δεκ"
]

plt.figure(figsize=(12, 6))
plt.bar(months[:len(monthly_costs)], monthly_costs, color="skyblue")
plt.title("Συνολικό Κόστος ανά Μήνα με χωρητικοτητα μπαταριας 5kw")
plt.xlabel("Μήνας")
plt.ylabel("Κόστος (€)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# 📊 Διάγραμμα Κόστους για διαφορετικες χωρητικοτητες
#plt.figure(figsize=(8, 5))
#plt.plot(battery_capacities, average_daily_costs, marker='o', linewidth=2, color='orange')
#plt.title("Σύγκριση Κόστους για Διαφορετικές Χωρητικότητες Μπαταρίας")
#plt.xlabel("Χωρητικότητα Μπαταρίας (kWh)")
#plt.ylabel("Μέσο Ημερήσιο Κόστος (€)")
#plt.grid(True)
#plt.xticks(battery_capacities)
#plt.tight_layout()
#plt.show()

#Διαγραμμα ροων ενεργειας και τιμων


fig, ax1 = plt.subplots(figsize=(16, 8))

## Rolling average για μείωση θορύβου (π.χ. παράθυρο 4 χρονικών βημάτων)
window = 4
for col, label, color, style in [
    ("Ppv_d", "PV ➔ Load", "gold", "-"),
    ("Ppv_b", "PV ➔ Battery", "orange", "--"),
    ("Ppv_g", "PV ➔ Grid", "red", ":"),
    ("Pb_d", "Battery ➔ Load", "green", "-"),
   ("Pb_g", "Battery ➔ Grid", "purple", "--"),
    ("Pg_d", "Grid ➔ Load", "brown", "-."),
    ("Pg_b", "Grid ➔ Battery", "grey", ":"),
    ("Pb_c", "Battery Charge", "pink", "--"),
    ("Pdisc", "Battery Discharge", "cyan", "-"),
    ("Pg_import", "Grid import", "blue", "-"),
    ("Pg_export", "Grid export", "magenta", "--"),
]:
    ax1.plot(
        day_df["timestep"],
        day_df[col].rolling(window, min_periods=1).mean(),
        label=label,
        color=color,
        linestyle=style,
        linewidth=2 if col in ["Ppv_d", "Pb_d", "Pg_import"] else 1
    )

ax1.set_xlabel("Χρονικό βήμα (15λεπτα)")
ax1.set_ylabel("Ενέργεια (kWh)")
ax1.grid(True, linestyle="--", alpha=0.5)

## Δεξιός άξονας για τιμές
ax2 = ax1.twinx()
ax2.plot(day_df["timestep"], day_df["BuyPrice"], label="Τιμή Αγοράς (€/kWh)", color="black", linestyle="--", linewidth=2)
ax2.plot(day_df["timestep"], day_df["SellPrice"], label="Τιμή Πώλησης (€/kWh)", color="magenta", linestyle=":", linewidth=2)
ax2.set_ylabel("Τιμή (€/kWh)")
ax2.tick_params(axis='y', labelcolor='black')

## Συνένωση υπομνημάτων
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
fig.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=10, title="Ροές & Τιμές")

plt.title("Όλες οι ροές ενέργειας και τιμές αγοράς/πώλησης - Ημέρα {day+1} ")
plt.tight_layout()
plt.show()


# Διαγραμμα κοστος ανα μερα
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(total_daily_costs) + 1), total_daily_costs, marker='o')
plt.title("Κόστος ανά Μέρα (Deterministic Model)")
plt.xlabel("Ημέρα")
plt.ylabel("Κόστος (€)")
plt.grid(True)
plt.tight_layout()
plt.show()