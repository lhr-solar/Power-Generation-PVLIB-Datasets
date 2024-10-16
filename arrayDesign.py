from pvlib import pvsystem, singlediode
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from scipy.constants import e as qe, k as kB

# For simplicity, use cell temperature of 25C for all calculations.
# kB is J/K, qe is C=J/V
# kB * T / qe -> V
Vth = kB * (273.15+25) / qe

cell_parameters = {
    'I_L_ref': 8.24,
    'I_o_ref': 2.36e-9,
    'a_ref': 1.3*Vth,
    'R_sh_ref': 1000,
    'R_s': 0.00181,
    'alpha_sc': 0.0042,
    'breakdown_factor': 2e-3,
    'breakdown_exp': 3,
    'breakdown_voltage': -15,
}

def simulate_full_curve(cell_params, Geff, Tcell, ivcurve_pnts=1000):
        # adjust the reference parameters according to the operating
    # conditions using the De Soto model:
    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
        Geff,
        Tcell,
        alpha_sc=cell_params['alpha_sc'],
        a_ref=cell_params['a_ref'],
        I_L_ref=cell_params['I_L_ref'],
        I_o_ref=cell_params['I_o_ref'],
        R_sh_ref=cell_params['R_sh_ref'],
        R_s=cell_params['R_s'],
    )
    sde_args = {
        'photocurrent': IL,
        'saturation_current': I0,
        'resistance_series': Rs,
        'resistance_shunt': Rsh,
        'nNsVth': nNsVth
    }
    curve_info = pvsystem.singlediode(method='lambertw', **sde_args)
    ivcurve_v = np.linspace(0., curve_info['v_oc'], ivcurve_pnts)
    ivcurve_i = pvsystem.i_from_v(voltage=ivcurve_v, method='lambertw', **sde_args)

    return pd.DataFrame({
        'i': ivcurve_i,
        'v': ivcurve_v
    })

def calcMPP_IscVocFF(Isys, Vsys):
    """from PVmismatch"""
    Psys = Isys * Vsys
    mpp = np.argmax(Psys)
    if Psys[mpp] == 0:
        Imp, Vmp, Pmp, Isc, Voc, FF = 0, 0, 0, 0, 0, 0
    else:
        P = Psys[mpp - 1:mpp + 2]
        V = Vsys[mpp - 1:mpp + 2]
        I = Isys[mpp - 1:mpp + 2]

        if any(P) == 0 or any(V) == 0 or any(I) == 0:
            Imp, Vmp, Pmp, Isc, Voc, FF = 0, 0, 0, 0, 0, 0
        else:
            # calculate derivative dP/dV using central difference
            dP = np.diff(P, axis=0)  # size is (2, 1)
            dV = np.diff(V, axis=0)  # size is (2, 1)
            if any(dP) == 0 or any(dV) == 0:
                Imp, Vmp, Pmp, Isc, Voc, FF = 0, 0, 0, 0, 0, 0
            else:
                # Pv = dP / dV  # size is (2, 1)
                Pv = np.divide(dP, dV, out=np.zeros_like(dP), where=dV!=0)
                # dP/dV is central difference at midpoints,
                Vmid = (V[1:] + V[:-1]) / 2.0  # size is (2, 1)
                Imid = (I[1:] + I[:-1]) / 2.0  # size is (2, 1)
                # interpolate to find Vmp

                Vmp = (-Pv[0] * np.diff(Vmid, axis=0) / np.diff(Pv, axis=0) + Vmid[0]).item()
                Imp = (-Pv[0] * np.diff(Imid, axis=0) / np.diff(Pv, axis=0) + Imid[0]).item()
                # calculate max power at Pv = 0
                Pmp = Imp * Vmp
                # calculate Voc, current must be increasing so flipup()
                # Voc = np.interp(np.float64(0), np.flipud(Isys),
                #                 np.flipud(Vsys))
                # Isc = np.interp(np.float64(0), Vsys, Isys)  # calculate Isc
                # FF = Pmp / Isc / Voc
    return dict(zip(['imp', 'vmp', 'pmp'], [Imp, Vmp, Pmp]))

def combine_series(dfs):
    df1 = dfs[0]
    imin = df1['i'].min()
    imax = df1['i'].max()
    i = np.linspace(imin, imax, 1000)
    v = 0
    for df2 in dfs:
        v_cell = interpolate(df2, i)
        v += v_cell
    return pd.DataFrame({'i': i, 'v': v})

def interpolate(df, i):
    f_interp = interp1d(np.flipud(df['i']), np.flipud(df['v']), kind='linear',
                        fill_value='extrapolate')
    return f_interp(i)

def plot_iv_curves(dfs, labels, title, yadjust=[0,0,3], legend=True):
    """
    Plot the Power-Voltage (PV) curve.
    Args:
    - dfs: list of pandas DataFrames containing 'i' (current) and 'v' (voltage) columns.
    - labels: list of labels for the curves.
    - title: title of the plot.
    """
    fig, axes = plt.subplots(sharey=True, figsize=(5, 3))
    for df, label in zip(dfs, labels):
        mpp_dict = calcMPP_IscVocFF(df['i'].values, df['v'].values)
        axes.plot(df['v'], df['i'], label=label)
        axes.set_xlim([0, df['v'].max()*3])
        axes.set_ylim([0, df['i'].max()*1.5])
        axes.plot([mpp_dict['vmp']], [mpp_dict['imp']], ls='', marker='o', c='k')
    
    axes.set_ylabel('current [A]')
    axes.set_xlabel('voltage [V]')
    fig.suptitle(title)
    fig.tight_layout()
    axes.legend()
    return axes

def plot_pv_curves(dfs, labels, title, yadjust=[0,0,3], legend=True):
    """
    Plot the Power-Voltage (PV) curve.
    Args:
    - dfs: list of pandas DataFrames containing 'i' (current) and 'v' (voltage) columns.
    - labels: list of labels for the curves.
    - title: title of the plot.
    """
    fig, axes = plt.subplots(sharey=True, figsize=(5, 3))
    for df, label in zip(dfs, labels):
        power = df['i'] * df['v']  # Calculate power
        df['p'] = power            # Add power column to the dataframe (optional, for debugging or future use)
        axes.plot(df['v'], power, label=label)
        axes.set_xlim([0, df['v'].max()*1.5])
        axes.set_ylim([0, df['p'].max()*1.5])
    
    axes.set_ylabel('power [W]')
    axes.set_xlabel('voltage [V]')
    fig.suptitle(title)
    fig.tight_layout()
    axes.legend()
    return axes

module_irrad = np.array(
    [
        [800, 800, 800, 800, 800, 800],
        [800, 800, 800, 800, 800, 800],
        [800, 800, 800, 800, 800, 800],
        [800, 800, 800, 800, 800, 800],
        [300, 800, 800, 800, 800, 800],
        [290, 285, 800, 800, 800, 800],
        [275, 280, 800, 800, 800, 800],
        [250, 260, 800, 800, 800, 800]

        # [300, 800, 800, 800, 800, 800],
        # [300, 800, 800, 800, 800, 800],
        # [300, 800, 800, 800, 800, 800],
        # [300, 800, 800, 800, 800, 800],
        # [300, 300, 800, 800, 800, 800],
        # [290, 285, 800, 800, 800, 800],
        # [275, 280, 800, 800, 800, 800],
        # [250, 260, 800, 800, 800, 800]
    ]
)

module_bypass = np.array(
    [
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2]
    ]
)

module_rows = module_irrad.shape[0]
module_cols = module_irrad.shape[1]

aa = np.arange(0, module_cols)
bb = np.arange(0, module_rows)

a, b = np.meshgrid(aa, bb)
module_idx = np.zeros(a.shape, dtype='int,int')
module_idx['f0'] = b
module_idx['f1'] = a

## Standard Conditions
noct_module_irrad = (module_irrad * 0) + 800
t_air = 20
NOCT = 42.5

module_iv = {}

for n,idx in enumerate(module_idx.flatten()):
        row = idx[0]
        col = idx[1]
        

        Gcell = module_irrad[row,col]
        Tcell = t_air + (((NOCT - 20) / 80) * (Gcell/10)) # standard NOCT temperature equation

        res = simulate_full_curve(
                cell_parameters,
                Gcell,
                Tcell)
        module_iv[f"{row},{col}"] = res
        
submodule_curves = []
for diode in [0,1,2]:
    submodule_idx = module_idx[module_bypass==diode]
    submodule_curve = combine_series([module_iv[f"{row},{col}"] for row,col in submodule_idx])
    submodule_curves.append(submodule_curve.clip(lower=-0.5))
    

module_iv_curve = combine_series(submodule_curves)
mpp_dict = calcMPP_IscVocFF(module_iv_curve["i"].values, module_iv_curve['v'].values)

plot_iv_curves([module_iv_curve],
                labels=["STC Module"],
                title='Module-level forward-biased IV curves',legend=False)
plot_pv_curves([module_iv_curve],
                labels=["STC Module"],
                title='Module-level forward-biased PV curves',legend=False)

plt.show()