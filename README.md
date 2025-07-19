# CROP_SUITABILITY
Contains codes for crop suitability
[source, ipython3]

### IMPORT LIBRARIES
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# INPUT FILES FOR EACH CROP
crops = {
    "Maize":      ("maize2_best_hist-ensmean_1971-2000.nc",
                   "maize2_best_ssp245-ensmean_2061-2090.nc",
                   "maize2_best_ssp585-ensmean_2061-2090.nc",
                   "maize2_best_G6sulfur-ensmean_2061-2090.nc"),
    "Pearlmillet":("Pearlmillet_best_hist-ensmean_1971-2000.nc",
                   "Pearlmillet_best_ssp245-ensmean_2061-2090.nc",
                   "Pearlmillet_best_ssp585-ensmean_2061-2090.nc",
                   "Pearlmillet_best_G6sulfur-ensmean_2061-2090.nc"),
    "Sorghum":    ("sorghum1_best_hist-ensmean_1971-2000.nc",
                   "sorghum1_best_ssp245-ensmean_2061-2090.nc",
                   "sorghum1_best_ssp585-ensmean_2061-2090.nc",
                   "sorghum1_best_G6sulfur-ensmean_2061-2090.nc"),
    "Cassava":    ("cassava1_best_hist-ensmean_1971-2000.nc",
                   "cassava1_best_ssp245-ensmean_2061-2090.nc",
                   "cassava1_best_ssp585-ensmean_2061-2090.nc",
                   "cassava1_best_G6sulfur-ensmean_2061-2090.nc"),
    "Cowpea":     ("Cowpea_best_hist-ensmean_1971-2000.nc",
                   "Cowpea_best_ssp245-ensmean_2061-2090.nc",
                   "Cowpea_best_ssp585-ensmean_2061-2090.nc",
                   "Cowpea_best_G6sulfur-ensmean_2061-2090.nc")
}

##DEFINE CLIMATE ZONES
zones = ["Guinea Coast", "Savannah", "Sahel"]
lat_bounds = {
    "Guinea Coast": (0, 4),
    "Savannah":      (4, 8),
    "Sahel":         (8, 11),
}
lon_slice = (-20, 20)

###COLOUR DEFINITION FOR THE SUITABILITY
colors = {"SSP245": "#55A868", "SSP585": "#C44E52", "G6SUL": "#4C72B0"}
width = 0.25

# COMPUTE CHANGE IN SUITABILITY FOR EACH CROP
delta_data = {}
for crop, (hist_f, s45_f, s85_f, g6_f) in crops.items():
    hist = xr.open_dataset(hist_f, decode_times=False)["b3suitv"]
    s45  = xr.open_dataset(s45_f, decode_times=False)["b3suitv"]
    s85  = xr.open_dataset(s85_f, decode_times=False)["b3suitv"]
    g6   = xr.open_dataset(g6_f,  decode_times=False)["b3suitv"]

    # normalize historical spatially
    hmin, hmax = hist.min(("lat","lon")), hist.max(("lat","lon"))
    hn = (hist - hmin) / (hmax - hmin)
    d45 = (s45 - hn).mean("time").sel(lon=slice(*lon_slice))
    d85 = (s85 - hn).mean("time").sel(lon=slice(*lon_slice))
    dg6 = (g6  - hn).mean("time").sel(lon=slice(*lon_slice))
    arrs = {"SSP245": [], "SSP585": [], "G6SUL": []}
    for zone in zones:
        lo, hi = lat_bounds[zone]
        arrs["SSP245"].append(np.ravel(d45.sel(lat=slice(lo, hi)).values))
        arrs["SSP585"].append(np.ravel(d85.sel(lat=slice(lo, hi)).values))
        arrs["G6SUL"].append(np.ravel(dg6.sel(lat=slice(lo, hi)).values))
    for scen in arrs:
        arrs[scen] = [a[~np.isnan(a)] for a in arrs[scen]]
    delta_data[crop] = arrs


# PLOTING FUNCTION
fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=True, constrained_layout=True)
axs_flat = axs.ravel()
x = np.arange(len(zones))
for idx, (crop, arrs) in enumerate(delta_data.items()):
    ax = axs_flat[idx]
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    # plot each scenario as offset boxplots
    for i, scen in enumerate(["SSP245", "SSP585", "G6SUL"]):
        ax.boxplot(
            arrs[scen],
            positions = x + (i-1)*width,
            widths    = width,
            notch     = True,
            patch_artist = True,
            showmeans   = True,
            meanprops   = dict(marker="D", markeredgecolor="black", markerfacecolor="white"),
            boxprops    = dict(facecolor=colors[scen], edgecolor="black"),
            whiskerprops= dict(color="black"),
            capprops    = dict(color="black"),
            medianprops = dict(color="firebrick")
        )
    # add significance
    for xi in x:
        y = max(np.max(arrs[s][xi]) for s in arrs) + 0.02
        p1 = stats.wilcoxon(arrs["SSP245"][xi], arrs["SSP585"][xi]).pvalue
        p2 = stats.wilcoxon(arrs["SSP585"][xi], arrs["G6SUL"][xi]).pvalue
        star = lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        ax.text(xi-width/2, y, star(p1), ha="center", fontsize=10)
        ax.text(xi+width/2, y, star(p2), ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(zones, fontsize=12)
    ax.set_title(crop, fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

# 6th PANEL: LEGEND
legend_ax = axs_flat[5]
legend_ax.axis("off")
handles = [plt.Line2D([0],[0], color=colors[s], lw=8) for s in colors]
legend_ax.legend(handles, list(colors.keys()),
                 loc="upper left", ncol=3, frameon=False,
                 title="Scenario", title_fontsize=12, fontsize=12)
axs[0,0].set_ylabel("Δ Suitability", fontsize=14)

# Save the plot
plt.savefig("Box_plot_crop_suit.png", dpi=600)
plt.show()

----


+*Out[2]:*+
----
![png](output_0_0.png)
----


+*In[5]:*+
[source, ipython3]
----
import os
###begin
# Define suitability categories and bounds
suitability_classes = ["Unsuitable", "Marginal", "Suitable", "Highly Suitable"]
suitability_bounds = [0, 0.24, 0.49, 0.74, 1.0]
region = dict(lon=slice(-20, 20), lat=slice(0, 20))  #for West Africa
crops = ["maize2", "sorghum1", "Pearlmillet", "cassava1", "Cowpea"]
scenario_order = ["Hist", "SSP245", "SSP585", "G6SUL"]
scenario_files = {
    "Hist": "hist-ensmean_1971-2000.nc",
    "SSP245": "ssp245-ensmean_2061-2090.nc",
    "SSP585": "ssp585-ensmean_2061-2090.nc",
    "G6SUL": "G6sulfur-ensmean_2061-2090.nc"
}
scenario_colors = {
    "Hist": "purple",
    "SSP245": "#1a9850", # green
    "SSP585": "red", # red
    "G6SUL": "#91bfdb"   # lightblue
}
bar_width = 0.20

###defining function
def load_suitability(crop, scenario):
    path = f"{crop}_best_{scenario_files[scenario]}"
    if not os.path.exists(path):
        print(f"Missing: {path}")
        return None
    ds = xr.open_dataset(path, decode_times=False)["b3suitv"]
    return ds.sel(**region)

def compute_fraction(arr2d):
    arr = arr2d.values
    valid = ~np.isnan(arr)
    total = valid.sum()
    fracs = []
    for low, high in zip(suitability_bounds[:-1], suitability_bounds[1:]):
        mask = (arr >= low) & (arr < high)
        fracs.append((mask & valid).sum() / total * 100 if total > 0 else 0)
    return np.array(fracs)

def compute_stats(data3d):
    ts = [compute_fraction(data3d.isel(time=t)) for t in range(data3d.sizes["time"])]
    arr = np.vstack(ts)
    return arr.mean(axis=0), arr.std(axis=0)
results = {}
for crop in crops:
    crop_res = {}
    for scen in scenario_order:
        da = load_suitability(crop, scen)
        if da is not None:
            m, s = compute_stats(da)
            crop_res[scen] = (m, s)
    results[crop] = crop_res

#Plotting function
fig, axes = plt.subplots(2, 3, figsize=(22, 14), sharey=True)
fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.07, wspace=0.15, hspace=0.3)
axes = axes.flatten()
x = np.arange(len(suitability_classes))

for idx, (ax, crop) in enumerate(zip(axes, crops + [None])):
    if crop is None:
        
        #Legend inside 6th subplot
        ax.axis('off')  # keep subplot for layout, hide ticks
        handles = [
            ax.bar(0, 0, color=scenario_colors[scen], edgecolor='black', label=scen)[0]
            for scen in scenario_order
        ]
        ax.legend(
            handles,
            scenario_order,
            loc='center',
            frameon=True,
            fontsize=14,
            framealpha=1,
            edgecolor='black',
            ncol=2,
            title='Scenario',
            title_fontsize=14
        )
        continue
    data = results[crop]
    for i, scen in enumerate(scenario_order):
        if scen in data:
            mean_frac, std_frac = data[scen]
            ax.bar(
                x + (i - 1.5) * bar_width,
                mean_frac,
                bar_width,
                yerr=std_frac,
                capsize=5,
                label=scen,
                color=scenario_colors[scen],
                edgecolor='black',
                linewidth=1.4
            )
    title = crop.replace('1', '').replace('2', '').replace('millet', ' Millet').capitalize()
    ax.set_title(f"{title} Suitability", fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(suitability_classes, rotation=45, fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='y', labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
axes[0].set_ylabel('Fraction of Land (%)', fontsize=16, fontweight='bold')

# save figure
plt.savefig('suitability_fractions_plots.png', dpi=500, bbox_inches='tight')
plt.show()

----


+*Out[5]:*+
----
![png](output_1_0.png)
----


+*In[6]:*+
[source, ipython3]
----
# for the suitability
import xarray as xr
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec
import pymannkendall as mk
from string import ascii_lowercase
global_hatch_width = 1.5
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["hatch.linewidth"] = global_hatch_width

##colour bound setup
bounds = [0, 0.24, 0.49, 0.74, 1.0]
colors = ["#8B4513", "#FFD700", "#ADFF2F", "#006400"]
labels = ["Unsuitable", "Marginal", "Suitable", "Very Suitable"]
cmap_suit = ListedColormap(colors)
norm_suit = BoundaryNorm(bounds, cmap_suit.N)

##mannkendal function
def mann_kendall_significance(data_array, alpha=0.01):
    _, lat_len, lon_len = data_array.shape
    p_values = np.full((lat_len, lon_len), np.nan)
    for i in range(lat_len):
        for j in range(lon_len):
            series = data_array[:, i, j].values
            if np.isnan(series).all():
                continue
            try:
                result = mk.original_test(series)
                p_values[i, j] = result.p
            except:
                continue
    return p_values < alpha
def load_and_process_suitability(hist_file, rcp85_file, sai_file):
    ds_hist = xr.open_dataset(hist_file, decode_times=False)["b3suitv"]
    ds_rcp = xr.open_dataset(rcp85_file, decode_times=False)["b3suitv"]
    ds_sai = xr.open_dataset(sai_file, decode_times=False)["b3suitv"]
    combined_proj = xr.concat([ds_rcp, ds_sai], dim="time")
    hist_norm = (ds_hist - ds_hist.min(dim="time")) / (
        ds_hist.max(dim="time") - ds_hist.min(dim="time"))
    hist_mean = hist_norm.mean(dim="time")
    diff_rcp = (ds_rcp - hist_norm).mean(dim="time")
    diff_sai = (ds_sai - hist_norm).mean(dim="time")
    diff_both = (ds_sai - ds_rcp).mean(dim="time")
    return hist_mean, diff_rcp, diff_sai, diff_both, combined_proj

##import data
data_inputs = {
    "Cassava": (
        "cassava1_best_hist-ensmean_1971-2000.nc",
        "cassava1_best_ssp585-ensmean_2061-2090.nc",
        "cassava1_best_G6sulfur-ensmean_2061-2090.nc"
    ),
    "Cowpea": (
        "Cowpea_best_hist-ensmean_1971-2000.nc",
        "Cowpea_best_ssp585-ensmean_2061-2090.nc",
        "Cowpea_best_G6sulfur-ensmean_2061-2090.nc"
    )
}
 #"Maize": ("maize2_best_hist-ensmean_1971-2000.nc","maize2_best_ssp585-ensmean_2061-2090.nc", "maize2_best_G6sulfur-ensmean_2061-2090.nc" ),
    #"Pearlmillet": ("Pearlmillet_best_hist-ensmean_1971-2000.nc", "Pearlmillet_best_ssp585-ensmean_2061-2090.nc", "Pearlmillet_best_G6sulfur-ensmean_2061-2090.nc"),
    #"Sorghum": ("sorghum1_best_hist-ensmean_1971-2000.nc", "sorghum1_best_ssp585-ensmean_2061-2090.nc", "sorghum1_best_G6sulfur-ensmean_2061-2090.nc"    )

datasets = {crop: load_and_process_suitability(*files)
    for crop, files in data_inputs.items()}

##plotting
fig = plt.figure(figsize=(24, 8), dpi=900)
gs = gridspec.GridSpec(nrows=2, ncols=4, hspace=0.01, wspace=0.04, figure=fig)
titles = ["Hist", "SSP585 – Hist", "G6SUL – Hist", "G6SUL – SSP585"]
extent = [-20, 20, 0, 20]
panel_labels = [f"({l})" for l in ascii_lowercase]
for row_idx, (crop, (hist, d1, d2, d3, combined_proj)) in enumerate(datasets.items()):
    sig_mask = mann_kendall_significance(combined_proj)
    mask_da = xr.DataArray(
        sig_mask.astype(float),
        coords={"lat": combined_proj.lat, "lon": combined_proj.lon},
        dims=["lat", "lon"],
    )
    for col_idx, (title, data) in enumerate(zip(titles, [hist, d1, d2, d3])):
        ax = fig.add_subplot(gs[row_idx, col_idx], projection=ccrs.PlateCarree())
        ax.set_extent(extent)
        ax.add_feature(cfeature.COASTLINE, linewidth=2.5)
        ax.add_feature(cfeature.BORDERS, linewidth=2.0)
        ax.gridlines(draw_labels=False, linestyle="--", linewidth=1.2,
                     color="gray", alpha=0.7)
        lon = data.lon.values
        lat = data.lat.values
        vmax = np.abs(data).max()
        if col_idx == 0:
            im = ax.pcolormesh(
                lon, lat, data,
                cmap=cmap_suit, norm=norm_suit,
                shading="auto", edgecolors='face', antialiased=False,
                transform=ccrs.PlateCarree()
            )
        else:
            im = ax.pcolormesh(
                lon, lat, data,
                cmap="PuOr", vmin=-vmax, vmax=vmax,
                shading="auto", edgecolors='face', antialiased=False,
                transform=ccrs.PlateCarree()
            )
            mask_i = mask_da.interp(lat=data.lat, lon=data.lon)
            hatch = ma.masked_where(mask_i.values < 0.5, np.ones_like(mask_i.values))
            cf = ax.contourf(
                lon, lat, hatch,
                levels=[0.5, 1.5],
                hatches=["//"], colors='none',
                transform=ccrs.PlateCarree(), zorder=10
            )
            for coll in cf.collections:
                coll.set_linewidth(global_hatch_width)

        ax.set_title(title, fontsize=18, fontweight="bold", pad=10)
        if col_idx == 0:
            ax.set_ylabel(crop, fontsize=20, fontweight="bold", labelpad=12)

        # Only show longitude ticks on last subplot
        if row_idx == 1 and col_idx == 3:
            ax.set_xticks(np.arange(extent[0], extent[1]+1, 10), crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(
                lambda x, _: f"{abs(int(x))}°{'E' if x > 0 else 'W'}"
            )
        else:
            ax.set_xticks([])

        # Only show latitude ticks on first column
        if col_idx == 0:
            ax.set_yticks(np.arange(extent[2], extent[3]+1, 5), crs=ccrs.PlateCarree())
            ax.yaxis.set_major_formatter(lambda y, _: f"{int(y)}°N")
        else:
            ax.set_yticks([])

        ax.tick_params(axis="both", labelsize=14, length=8)

        label_idx = row_idx * 4 + col_idx
        ax.text(
            0.02, 0.02, panel_labels[label_idx],
            transform=ax.transAxes,
            fontsize=20, fontweight="bold",
            va="bottom", ha="left", backgroundcolor="white"
        )
centers = [(bounds[i] + bounds[i+1]) / 2 for i in range(len(bounds)-1)]

##legends
cax1 = fig.add_axes([0.20, 0.04, 0.6, 0.02])
cb1 = fig.colorbar(
    plt.cm.ScalarMappable(cmap=cmap_suit, norm=norm_suit),
    cax=cax1, orientation="horizontal", ticks=bounds[:-1]
)
cb1.set_ticks(centers)
cb1.set_label("Suitability", fontsize=22, fontweight="bold", labelpad=20)
cb1.ax.tick_params(labelsize=20, width=2, length=10)
cb1.set_ticklabels(labels)

v = np.abs(next(iter(datasets.values()))[1]).max()
cax2 = fig.add_axes([0.935, 0.10, 0.02, 0.75])
cb2 = fig.colorbar(
    plt.cm.ScalarMappable(norm=plt.Normalize(-v, v), cmap="PuOr"),
    cax=cax2, orientation="vertical", extend="both"
)
cb2.set_label("Δ Suitability", fontsize=22, fontweight="bold", labelpad=20)
cb2.ax.tick_params(labelsize=18, width=2, length=10)

# save figure
plt.savefig("Cassava_suit_cowpea_585_WA.png", bbox_inches="tight")
plt.show()
