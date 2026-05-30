# Hf2Mo2Ta48W48: Alloy 0
# Mole fraction
X_Hf = 0.02
X_Mo = 0.02
X_Ta = 0.48
X_W = 0.48

# Boiling temperature (K)
Tb_Hf = 4876.15 
Tb_Mo = 4912.15
Tb_Ta = 5731.15
Tb_W = 5828.15

# Latent heat of vaporization (J/mol)
Lvap_Hf = 648000
Lvap_Mo = 598000
Lvap_Ta = 753000
Lvap_W = 774000

# Molecular weight (g/mol)
MW_0 = 180.586792

# Using rule of mixtures, find boiling point and heat of vaporization of HEA
Tb_0 = X_Hf*Tb_Hf + X_Mo*Tb_Mo + X_Ta*Tb_Ta + X_W*Tb_W
print("Boiling point:", Tb_0, "K")

Lvap_0 = X_Hf*Lvap_Hf + X_Mo*Lvap_Mo + X_Ta*Lvap_Ta + X_W*Lvap_W
print("Latent heat of vaporization:", Lvap_0, "J/mol")

# Effective heat capacity (J/(kg*K))
# Cp_eff = Cp + Lvap/(Tb-Tliq)
Cp_eff = 251.6173225 + (Lvap_0/(MW_0/1000))/(Tb_0-3454.846467) #ORIGINAL --> OVERCORRECTING
#Cp_eff = 251.6173225 + (Lvap_0/(MW_0/1000))/(300)
print("Effective Cp:", Cp_eff, "J/kgK")

# Sofia's version of effective heat capacity
# Cp = (H_liquidus - H_RT)/(T_liquidus - T_RT)
H_RT = -7733.89863 #J/mol
H_liquidus = 1.40774E5 #J/mol		
T_RT = 298 #K
T_liquidus = 3455 #K, just above liquidus
Cp_eff2 = (H_liquidus - H_RT)/(T_liquidus - T_RT) #J/mol*K
MW = 180.59260 #g/mol
Cp_eff2 = Cp_eff2 / MW * 1000
print("Effective Cp using thermocalc:", Cp_eff2)

# Sensitivity run: Sofia's method of Cp with boiling temp instead
# Cp = (H_boiling - H_RT)/(T_boiling - T_RT)
H_RT = -7733.89863 #J/mol
H_boiling = 6.24697E5 #J/mol		
T_RT = 298 #K
T_boiling = 5881 #K, when gas mole fraction = 0.5
Cp_eff3 = (H_boiling - H_RT)/(T_boiling - T_RT) #J/mol*K
MW = 180.59260 #g/mol
Cp_eff3 = Cp_eff3 / MW * 1000
print("Effective Cp using boiling enthalpy:", Cp_eff3)

# Effective heat capacity, using THERMOCALC VALUES (J/(kg*K))
# Cp_eff = Cp + Lvap/(Tb-Tliq)
Lvap0 = 0
Cp_eff = 251.6173225 + (Lvap_0/(MW_0/1000))/(Tb_0-3454.846467) #ORIGINAL --> OVERCORRECTING
#Cp_eff = 251.6173225 + (Lvap_0/(MW_0/1000))/(300)
print("Effective Cp:", Cp_eff, "J/kgK")