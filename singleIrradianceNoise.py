from pvlib import pvsystem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

module_params = {
    'alpha_sc': 0.004539, # temp coefficient of short circuit current
    'a_ref': 1.5, # diode ideality factor
    'I_L_ref': 9.5, # light-generated current at reference condtions
    'I_o_ref': 1e-10, # saturation current, higher is worse
    'R_sh_ref': 500, # shunt resistance of cell
    'R_s': 0.5, # total series resistance
    'EgRef': 1.121,
}

irradiances = np.linspace(200, 1000, num=50)
shaded_irradiances = irradiances * 0.5
temperatures = np.linspace(15, 75, num=50)

# irradiance_noise = np.random.normal(0, 0.52 * irradiances.mean(), size=irradiances.shape)
# temperature_noise = np.random.normal(0, 0.52 * temperatures.mean(), size=temperatures.shape)

# irradiances = irradiances + irradiance_noise
# temperatures = temperatures + temperature_noise 

# big dataset so lots of cases of irradiances and temperature
conditions = pd.DataFrame([(x,y) for x in irradiances for y in temperatures], columns=['Geff', 'Tcell'])
shaded_conditions = pd.DataFrame([(x,y) for x in shaded_irradiances for y in temperatures], columns=['Geff', 'Tcell'])

def gen_curves(conds):
    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
        conds['Geff'],
        conds['Tcell'],
        alpha_sc=module_params['alpha_sc'],
        a_ref=module_params['a_ref'],
        I_L_ref=module_params['I_L_ref'],
        I_o_ref=module_params['I_o_ref'],
        R_sh_ref=module_params['R_sh_ref'],
        R_s=module_params['R_s'],
        EgRef=module_params['EgRef'],
        dEgdT=-0.0002677
    )
    single_diode_params = {
        # can't add noise here because needs to be some minimum values (>0, etc.)
        # also these values are really small (10^-10) so not really going to work
        'photocurrent': IL,
        'saturation_current': I0,
        'resistance_shunt': Rsh,
        'resistance_series': Rs,
        'nNsVth': nNsVth
    }
    curve_info = pvsystem.singlediode(method='lambertw', **single_diode_params)
    return curve_info, single_diode_params


# IL = np.clip(IL, 0, None)  # Ensure IL is not negative
# I0 = np.clip(I0, 0, None)  # Ensure I0 is not negative
# Rs = np.clip(Rs, 0, None)  # Ensure Rs is not negative
# Rsh = np.clip(Rsh, 1e-3, None)  # Avoid too low values of Rsh
# nNsVth = np.clip(nNsVth, 0, None)  # Ensure nNsVth is not negative


org_curve, org_params = gen_curves(conditions)
shaded_curve, shaded_params = gen_curves(shaded_conditions)
org_v = pd.DataFrame(np.linspace(0, org_curve['v_oc'], 100))
shaded_v = pd.DataFrame(np.linspace(0, shaded_curve['v_oc'], 100))
#i = pd.DataFrame(np.linspace(0, curve_info['i_sc'], 100))
org_i = pd.DataFrame(pvsystem.i_from_v(voltage=org_v, method='lambertw', **org_params))
shaded_i = pd.DataFrame(pvsystem.i_from_v(voltage=shaded_v, method='lambertw', **shaded_params))
# just for clarity, add the noise to current that will propogate through to the power
# i_noise = np.random.normal(0, 0.02 * i.max(), size=i.shape)  # Add noise relative to power range
# i_noisy = i + i_noise

# combining shaded and unshaded
final_v = np.linspace(0, 40, 100)
interp_org_i = np.interp(final_v, org_v[0], org_i[0])
interp_shaded_i = np.interp(final_v, shaded_v[0], shaded_i[0])
final_i = np.minimum(interp_org_i, interp_shaded_i)

p = final_i * final_v
# idea 1, just add random noise directly to the power, would work but does not change the current/voltage for training
# noise = np.random.normal(0, 0.02 * p.max(), size=p.shape)  # Add noise relative to power range
# p_noisy = p + noise  # Noisy power

# new_max_power_idx = np.argmax(p[0])
# new_v_mp = v[0][new_max_power_idx]
# new_p_mp = p[0][new_max_power_idx]

plt.figure()
label = (
    "$G_{eff}$ " + f"{conditions.head(1)['Geff']} $W/m^2$\n"
    "$T_{cell}$ " + f"{conditions.head(1)['Tcell']} $\\degree C$"
)
plt.plot(org_v[0], org_i[0], label=label)
plt.plot(shaded_v[0], shaded_i[0], label=label)
plt.plot(final_v[0], final_i[0], label=label)
#plt.plot(final_v, p, label=label)
v_mp = shaded_curve['v_mp'][0]
i_mp = shaded_curve['i_mp'][0]
p_mp = shaded_curve['p_mp'][0]
# mark mppt
#plt.plot([v_mp], [p_mp], ls='', marker='o', c='k')
# plt.plot([new_v_mp], [new_p_mp], ls='', marker='x', c='k')

plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend(loc=(1.0,0))
plt.xlabel("voltage")
plt.ylabel("power")
plt.gcf().set_tight_layout(True)

plt.show()
