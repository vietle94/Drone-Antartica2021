from scipy import stats
from functools import reduce
import pandas as pd
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import fnmatch
import xarray as xr
from sklearn.metrics import r2_score
import copy
import matplotlib
%matplotlib qt

################################
# %%
################################
summary_dir = r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Antarctica\Viet/summary_plots/'
daily_dir = r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Antarctica\Viet/daily_plots/'

sensor_dir = r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Antarctica\Viet\combined_data\Daily_sensors/'
sensor_path = [x for x in glob.glob(sensor_dir + '*.csv') if 'cal_flight' not in x]

wind_data = pd.read_csv(
    r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Antarctica\Viet\combined_data\Wind/wind_merged.csv')
wind_data['datetime'] = pd.to_datetime(wind_data['datetime'])

weather_data = pd.read_csv(
    r'C:\Users\le\OneDrive - Ilmatieteen laitos\My files\drone_backpack\Antarctica\mavic_Matt/weather.csv')
weather_data = weather_data.rename({'date_time': 'datetime'}, axis='columns')
weather_data.datetime = pd.to_datetime(weather_data.datetime)
weather_data = weather_data.drop(['wdir_10m', 'ws_10m'], axis=1)

# %%
df = pd.DataFrame({})
for i, x in enumerate(sensor_path):
    df_ = pd.read_csv(x)
    df_['flight_ID'] = i
    df_.dropna(how='all', inplace=True)
    i_min = np.argmin(df_.press_bme_BME_BP3)
    df_['ascend'] = True
    df_.loc[i_min:, 'ascend'] = False
    df = df.append(df_)

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.reset_index(drop=True)

# %%
df_full = df.merge(wind_data, on='datetime', how='left')
df_full = df_full.merge(weather_data, on='datetime', how='left')

df_full[['wdir_3s', 'rh_1m', 'press_1m', 'press_surf_1m', 'glob_rad',
         'temp_1m', 'dewpoint', 'ws_3s', 'ws_gust',
         'voltage']] = df_full[['wdir_3s', 'rh_1m', 'press_1m', 'press_surf_1m', 'glob_rad',
                                'temp_1m', 'dewpoint', 'ws_3s', 'ws_gust',
                                'voltage']].fillna(method='backfill', limit=60)
df_full[['Altitude', 'Home Distance',
         'Wind Direction', 'Wind Speed']] = df_full[['Altitude', 'Home Distance',
                                                     'Wind Direction', 'Wind Speed']].fillna(method='ffill', limit=10)
df_full = df_full[df_full.datetime.dt.year != 1970]
df_full = df_full.reset_index(drop=True)
df_full = df_full[~pd.isnull(df_full['datetime'])]

# %%
bin_boundaries = [0.38, 0.46, 0.66, 0.915, 1.195, 1.465,
                  1.83, 2.535, 3.5, 4.5, 5.75, 7.25, 9, 11, 13, 15, 16.75]
dlog_bin = np.log10(bin_boundaries[1:]) - np.log10(bin_boundaries[:-1])

particle_size_colname = [x for x in df_full.columns if x.split(
    '_')[0].replace('.', '', 1).isdigit()]

total_volume = (3.66666*df_full['sampling_time_OPC_BP3'])
df_full['total_concentration'] = df_full.loc[:, particle_size_colname].sum(
    axis=1, min_count=1) / (total_volume)

for particle_size, each_dlog_bin in zip(particle_size_colname, dlog_bin):
    df_full[particle_size] = df_full[particle_size]/total_volume / each_dlog_bin

df_full.columns

##########################################################
# %% Compare bme and sht
##########################################################

fig, ax = plt.subplots(3, 2, figsize=(16, 9), constrained_layout=True)

for data_plot, plot_name, ax_row in zip([df_full, df_full[df_full['ascend'] == True],
                                         df_full[df_full['ascend'] == False]],
                                        ['full', 'ascending', 'descending'],
                                        ax):
    # fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for ax_, x, y in zip(ax_row.flatten(),
                         ['temp_bme_BME_BP3', 'rh_bme_BME_BP3'],
                         ['temp_sht_SHT_BP3', 'rh_sht_SHT_BP3']):

        ax_.plot(data_plot[x], data_plot[y], "+",
                 ms=5, mec="k", alpha=0.01)
        ax_.set_xlabel(x)
        ax_.set_ylabel(y)
        ax_.grid()
        ax_.set_title(plot_name, weight='bold')
        line = data_plot[[x, y]].copy()
        line = line.dropna()
        z = np.polyfit(line[x],
                       line[y], 1)

        y_hat = np.poly1d(z)(line[x])
        ax_.plot(line[x], y_hat, "r-", lw=1)
        text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(line[y], y_hat):0.3f}$"
        ax_.text(0.05, 0.95, text, transform=ax_.transAxes,
                 fontsize=10, verticalalignment='top')
        axline_min = np.min((np.min(data_plot[x]), np.min(data_plot[y])))
        axline_max = np.min((np.max(data_plot[x]), np.max(data_plot[y])))
        ax_.axline((axline_min, axline_min),
                   (axline_max, axline_max),
                   color='grey', linewidth=0.5, ls='--')
        ax_.grid()

    fig.savefig(summary_dir + '/bme_sht_.png', bbox_inches='tight')
    # plt.close()

##########################################################
# %% Particles parameters
##########################################################
for i, grp in df_full.groupby(['flight_ID']):
    grp = grp.reset_index(drop=True)
    grp[['datetime', 'press_bme_BME_BP3']] = grp[['datetime', 'press_bme_BME_BP3']].set_index(
        'datetime').rolling('10s').median().reset_index()

    # remove first couple of measurements
    grp_rm1hpa = grp[grp.press_bme_BME_BP3 < (np.nanmax(grp.press_bme_BME_BP3) - 1)].copy()
    bin_width = 5
    lower_bin = grp_rm1hpa.press_bme_BME_BP3.min() - grp_rm1hpa.press_bme_BME_BP3.min() % bin_width
    bins = np.arange(lower_bin, grp_rm1hpa.press_bme_BME_BP3.max()+bin_width, bin_width)
    labels = (bins[1:] + bins[:-1])/2
    fig1, ax = plt.subplots(3, 5, figsize=(16, 9), sharex='col', sharey='row')

    for grp_, flight_label, ax_ in zip([grp_rm1hpa.loc[grp_rm1hpa['ascend']],
                                        grp_rm1hpa.loc[~grp_rm1hpa['ascend']],
                                        grp_rm1hpa],
                                       ['ASCEND', 'DESCEND', 'ASCEND + DESCEND'],
                                       ax):

        grp_particle_plot = grp_.copy()
        grp_particle_plot['p_binned'] = pd.cut(
            grp_['press_bme_BME_BP3'], bins=bins, labels=labels, include_lowest=True)

        grp_particle_plot.dropna(axis=0, how='all', inplace=True)
        grp_particle_plot = grp_particle_plot.reset_index(drop=True)

        grp_particle_plot = grp_particle_plot.groupby('p_binned').mean()

        ax_[0].plot(grp_particle_plot.pm1_OPC_BP3, grp_particle_plot.index, label='PM1')
        ax_[0].plot(grp_particle_plot.pm25_OPC_BP3, grp_particle_plot.index, label='PM2.5')
        ax_[0].plot(grp_particle_plot.pm10_OPC_BP3, grp_particle_plot.index, label='PM10')
        ax_[0].set_ylabel(flight_label + '\nPressure bme')
        # ax_[0].set_xlabel('Mass concentration')
        ax_[0].legend()
        # ax_[0].set_xlim(left=0)

        ax_[1].plot(grp_particle_plot.total_concentration, grp_particle_plot.index)
        # ax_[1].set_xlabel('Total concentration')

        ax_[2].plot(grp_particle_plot.temp_bme_BME_BP3,
                    grp_particle_plot.index, label='temp_bme_BME_BP3')
        ax_[2].plot(grp_particle_plot.temp_sht_SHT_BP3,
                    grp_particle_plot.index, label='temp_sht_SHT_BP3')
        # ax_[2].set_xlabel('Temperature')
        ax_[2].legend()

        ax_[3].plot(grp_particle_plot.rh_bme_BME_BP3,
                    grp_particle_plot.index, label='rh_bme_BME_BP3')
        ax_[3].plot(grp_particle_plot.rh_sht_SHT_BP3,
                    grp_particle_plot.index, label='rh_sht_SHT_BP3')
        # ax_[3].set_xlabel('RH')
        ax_[3].legend()
        for i_, ax__ in enumerate(ax_):
            ax__.grid()
            # if i_ != 0:
            #     plt.setp(ax_.get_yticklabels(), visible=False)
        p = ax_[4].pcolormesh([float(x.split('_')[0]) for x in particle_size_colname], labels,
                              grp_particle_plot.loc[:, particle_size_colname])
        cbar = fig1.colorbar(p, ax=ax_[4])
        cbar.ax.set_ylabel('dN/dlogDp')
        plt.setp(ax_[4].get_yticklabels(), visible=False)
        ax_[4].set_xscale('log')

        # ax_[4].set_xticks((np.arange(len(particle_size_colname)) + 0.5)[::2])
        ax_[0].invert_yaxis()
    ax_[0].set_xlabel('Mass concentration')
    ax_[1].set_xlabel('Total concentration')
    ax_[2].set_xlabel('Temperature')
    ax_[3].set_xlabel('RH')
    ax_[4].set_xlabel('Particle size')
    fig1.savefig(daily_dir + str(grp.datetime.min()).replace(' ', '_').replace(':',
                 '-') + '_particle_profile.png', bbox_inches='tight')
    plt.close('all')


#####################################################################
# %% Meteorological parameters
#####################################################################
ref = pd.DataFrame({})
ref_wind = pd.DataFrame({})
for i, grp in df_full.groupby(['flight_ID']):
    grp = grp.reset_index(drop=True)
    grp[['datetime', 'press_bme_BME_BP3']] = grp[['datetime', 'press_bme_BME_BP3']].set_index(
        'datetime').rolling('10s').median().reset_index()
    fig2, ax = plt.subplots(3, 2, figsize=(16, 9), sharex=True)
    ax[0, 0].plot(grp.datetime, grp.press_bme_BME_BP3, '.', label='press_bme_BME_BP3')
    ax[0, 0].plot(grp.datetime, grp.press_1m, '.', label='press_1m_tower')
    ax[0, 0].set_ylabel('Pressure [hPa]')

    ax[0, 1].plot(grp.datetime, grp.Altitude, '.', label='Drone Altitude from wind')
    ax[0, 1].set_ylabel('Altitude [m]')

    ax[1, 0].plot(grp.datetime, grp.temp_bme_BME_BP3, '.', label='temp_bme_BME_BP3')
    ax[1, 0].plot(grp.datetime, grp.temp_sht_SHT_BP3, '.', label='temp_sht_SHT_BP3')
    ax[1, 0].plot(grp.datetime, grp.temp_1m, '.', label='temp_1m_tower')
    ax[1, 0].set_ylabel('Temperature [$^0C$]')

    ax[1, 1].plot(grp.datetime, grp.rh_bme_BME_BP3, '.', label='rh_bme_BME_BP3')
    ax[1, 1].plot(grp.datetime, grp.rh_sht_SHT_BP3, '.', label='rh_sht_SHT_BP3')
    ax[1, 1].plot(grp.datetime, grp.rh_1m, '.', label='rh_1m_tower')
    ax[1, 1].set_ylabel('Relative humidity [%]')

    ax[2, 0].plot(grp.datetime, grp['Wind Speed'], '.', label='Wind speed')
    ax[2, 0].plot(grp.datetime, grp['ws_3s'], '.', label='wind_3s_tower')
    ax[2, 0].set_ylabel('Wind Speed [m/s]')

    ax[2, 1].plot(grp.datetime, grp['Wind Direction'], '.', label='Wind direction')
    ax[2, 1].plot(grp.datetime, grp['wdir_3s'], '.', label='wind_direction_3s_tower')
    ax[2, 1].set_ylabel('Wind direction [$^0$]')

    for ax_ in ax.flatten():
        ax_.legend()
        ax_.grid()
    thres_pressure = 0.5
    grp['compare_tower_others'] = np.abs(grp.press_1m - grp.press_bme_BME_BP3) < thres_pressure
    grp['compare_tower_wind'] = np.abs(
        (grp.press_1m - 0.7) - grp.press_bme_BME_BP3) < thres_pressure
    ref = ref.append(grp[grp['compare_tower_others'] == True], ignore_index=True)
    ref_wind = ref_wind.append(grp[grp['compare_tower_wind'] == True],
                               ignore_index=True)

    for vline in grp[grp['compare_tower_others']].datetime:
        for ax_ in ax.flatten()[:-2]:
            ax_.axvline(x=vline, alpha=0.2)
    for vline in grp[grp['compare_tower_wind']].datetime:
        for ax_ in ax.flatten()[-2:]:
            ax_.axvline(x=vline, alpha=0.2, c='orange')
    fig2.savefig(daily_dir + str(grp.datetime.min()).replace(' ', '_').replace(':',
                 '-') + '_met_ts.png', bbox_inches='tight')
    print(str(grp.datetime.min()).replace(' ', '_').replace(':', '-'))
    plt.close('all')

# %%
fig, ax = plt.subplots(3, 4, figsize=(16, 9), sharex='col', sharey='col')
for data_plot, plot_name, ax_row in zip([ref[ref['ascend'] == True],
                                         ref[ref['ascend'] == False], ref],
                                        ['ASCEND', 'DESCEND', 'ASCEND + DESCEND'],
                                        ax):
    for ax_, x, y in zip(ax_row,
                         ['temp_1m', 'rh_1m',
                          'temp_1m', 'rh_1m'],
                         ['temp_bme_BME_BP3', 'rh_bme_BME_BP3',
                          'temp_sht_SHT_BP3', 'rh_sht_SHT_BP3']
                         ):

        ax_.plot(data_plot[x], data_plot[y], "+",
                 ms=5, mec="k", alpha=0.1)
        ax_.set_xlabel(x)
        ax_.set_ylabel(y)
        ax_.grid()
        ax_.set_title(plot_name, weight='bold')
        line = data_plot[[x, y]].copy()
        line = line.dropna()
        z = np.polyfit(line[x],
                       line[y], 1)

        y_hat = np.poly1d(z)(line[x])
        ax_.plot(line[x], y_hat, "r-", lw=1)
        text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(line[y], y_hat):0.3f}$"
        ax_.text(0.05, 0.95, text, transform=ax_.transAxes,
                 fontsize=10, verticalalignment='top')
        axline_min = np.min((np.min(data_plot[x]), np.min(data_plot[y])))
        axline_max = np.min((np.max(data_plot[x]), np.max(data_plot[y])))
        ax_.axline((axline_min, axline_min),
                   (axline_max, axline_max),
                   color='grey', linewidth=0.5, ls='--')
        ax_.grid()

fig.subplots_adjust(hspace=0.5, wspace=0.5)
fig.savefig(summary_dir + '/temp_rh.png', bbox_inches='tight')

# %%
fig, ax = plt.subplots(3, 2, figsize=(16, 9))
for data_plot, plot_name, ax_row in zip([ref_wind[ref_wind['ascend'] == True],
                                         ref_wind[ref_wind['ascend'] == False], ref_wind],
                                        ['ASCEND', 'DESCEND', 'ASCEND + DESCEND'],
                                        ax):
    for ax_, x, y in zip(ax_row,
                         ['wdir_3s', 'ws_3s'],
                         ['Wind Direction', 'Wind Speed']
                         ):

        ax_.plot(data_plot[x], data_plot[y], "+",
                 ms=5, mec="k", alpha=0.5)
        ax_.set_xlabel(x)
        ax_.set_ylabel(y)
        ax_.grid()
        ax_.set_title(plot_name, weight='bold')
        line = data_plot[[x, y]].copy()
        line = line.dropna()
        z = np.polyfit(line[x],
                       line[y], 1)

        y_hat = np.poly1d(z)(line[x])
        ax_.plot(line[x], y_hat, "r-", lw=1)
        text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(line[y], y_hat):0.3f}$"
        ax_.text(0.05, 0.95, text, transform=ax_.transAxes,
                 fontsize=10, verticalalignment='top')
        axline_min = np.min((np.min(data_plot[x]), np.min(data_plot[y])))
        axline_max = np.min((np.max(data_plot[x]), np.max(data_plot[y])))
        ax_.axline((axline_min, axline_min),
                   (axline_max, axline_max),
                   color='grey', linewidth=0.5, ls='--')
        ax_.grid()

fig.subplots_adjust(hspace=0.5, wspace=0.2)
fig.savefig(summary_dir + '/wind.png', bbox_inches='tight')
plt.close()
