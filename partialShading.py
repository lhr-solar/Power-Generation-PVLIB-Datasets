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

def sim_full_curve(cell_params, Geff, Tcell, ivcurve_pnts=1000):
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

def plot_curves(dfs, labels, title):
    """plot the forward- and reverse-bias portions of an IV curve"""
    fig, axes = plt.subplots(sharey=True, figsize=(5, 3))
    for df, label in zip(dfs, labels):
        df.plot('v', 'i', label=label, ax=axes)
        axes.set_xlim([0, df['v'].max()*1.5])
    axes.set_ylabel('current [A]')
    axes.set_xlabel('voltage [V]')
    fig.suptitle(title)
    fig.tight_layout()
    return axes

cell_curve_full_sun = sim_full_curve(cell_parameters, Geff=1000, Tcell=25)
cell_curve_shaded = sim_full_curve(cell_parameters, Geff=200, Tcell=25)
ax = plot_curves([cell_curve_full_sun, cell_curve_shaded],
                 labels=['Full Sun', 'Shaded'],
                 title='Cell-level forward-biased IV curves')

'''
To combine the individual cell IV curves and form a module's IV curve, the cells in each substring must be added in series. 
The substrings are in series as well, but with parallel bypass diodes to protect from reverse bias voltages. To add in series, the voltages for a given current are added. 
However, because each cell's curve is discretized and the currents might not line up, we align each curve to a common set of current values with interpolation.
'''
def interpolate(df, i):
    f_interp = interp1d(np.flipud(df['i']), np.flipud(df['v']), kind='linear',
                        fill_value='extrapolate')
    return f_interp(i)

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

# rather than simulating 72 cells, simulate 3 cells with each different type is sufficient to simulate module
# this function also simulates the bypass diodes in parallel with each substring 
#    - not sure if its relevant here b/c no reverse-bias
def simulate_module(cell_params, 
                    poa_direct, # direct irradiance that strikes the surface, comes straight from sun without being scattered by atmosphere
                    poa_diffuse, # diffuse irradiance, which is sunlight that has been scattered by molecules and particles
                    Tcell, 
                    shaded_fraction, 
                    cells_per_string=24,
                    strings=3):
    '''
    cell temp uniform across module
    shade assuming to be coming up from bottom of module, so affects all substrings equally
    substrings are "down and back" so number of cells per string is divided between two columns of cells
    '''
    nrow = cells_per_string // 2 # cells per column that are in full shadow
    nrow_full_shade = int(shaded_fraction * nrow)
    # fraction of shade in the border row
    partial_shade_fraction = 1 - (shaded_fraction * nrow - nrow_full_shade)
    df_lit = sim_full_curve( # df means dataframe
        cell_parameters,
        poa_diffuse + poa_direct,
        Tcell)
    df_partial = sim_full_curve(
        cell_parameters,
        poa_diffuse + partial_shade_fraction * poa_direct,
        Tcell)
    df_shaded = sim_full_curve(
        cell_parameters,
        poa_diffuse,
        Tcell)

    include_partial_cell = (shaded_fraction < 1)
    # understand what this does
    half_substring_curves = (
        [df_lit] * (nrow - nrow_full_shade - 1)
        + ([df_partial] if include_partial_cell else [])  # noqa: W503
        + [df_shaded] * nrow_full_shade  # noqa: W503
    )
    substring_curve = combine_series(half_substring_curves)
    substring_curve['v'] *= 2 # turn into whole strings
    # bypass_diode
    substring_curve['v'] = substring_curve['v'].clip(lower=-0.5)
    # no need to interpolate since we're just scaling voltage directly:
    substring_curve['v'] *= strings
    return substring_curve
    
kwargs = {
    'cell_params' : cell_parameters,
    'poa_direct': 800,
    'poa_diffuse': 200,
    'Tcell':  25
}
module_curve_full_sun = simulate_module(shaded_fraction=0, **kwargs)
module_curve_shaded = simulate_module(shaded_fraction=0.1, **kwargs)
ax = plot_curves([module_curve_full_sun, module_curve_shaded],
                 labels=['Full Sun', 'Shaded'],
                 title='Module-level forward-biased IV curves')

plt.show()