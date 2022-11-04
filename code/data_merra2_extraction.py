""" 
Check if host machine contains required packages to run this process. 
If packages are not available, this process will install them.
This process will get surface temperature, bias-corrected precipitation, 
cloud fraction, and surface wind speed from different MERRA-2 collections,
resample them to daily data and save to a csv file.
User can specify duration of data to extract in lines (67 - 71).
NOTE: This process also needs a NASA Earthdata account.
Please update line 46 with a username and password to proceed with execution.
 """

import sys
import subprocess
import pkg_resources

# Required packages to run this process.
required = {'xarray', 'netCDF4', 'dask', 'pandas'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print("Packages missing and will be installed: ", missing)
    python = sys.executable
    subprocess.check_call(
        [python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

################################
#  END OF PACKAGES VALIDATION  #
################################


##############################################
#  NASA Earthdata account credentials setup  #
##############################################

# get current user home directory to create necessary file for earthdata access.
from os.path import expanduser
home_dir = expanduser("~")

# create .netrc file to insert earthdata username and password
open(home_dir + '/.netrc', 'w').close()
open(home_dir + '/.urs_cookies', 'w').close()
subprocess.check_call(
    ['echo "machine urs.earthdata.nasa.gov login <username> password <password>" >> ' + home_dir + '/.netrc'], shell=True)

open(home_dir + '/.dodsrc', 'w').close()
subprocess.check_call(
    ['echo "HTTP.COOKIEJAR=' + home_dir + '/.urs_cookies" >> ' + home_dir + '/.dodsrc'], shell=True)
subprocess.check_call(
    ['echo "HTTP.NETRC=' + home_dir + '/.netrc" >> ' + home_dir + '/.dodsrc'], shell=True)

#####################################################
#  END OF NASA Earthdata account credentials setup  #
#####################################################


""" Extract MERRA-2 Hourly data for the month of January 2019 """
import pandas as pd
import dask
import netCDF4
import xarray as xr


# Time frame of MERRA-2 data to collect
year = '2019'
month_begin = '01'
month_end = '03'
day_begin = '01'
day_end = '31'

# MERRA-2 M2I1NXASM collection (hourly) to get Temp and Wind variables (T2M, V2M).
collection_shortname = 'M2I1NXASM'
collection_longname = 'inst1_2d_asm_Nx'
collection_number = 'MERRA2_400'
MERRA2_version = '5.12.4'


# OPeNDAP URL
url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/{}.{}/{}'.format(
    collection_shortname, MERRA2_version, year)
files_month = ['{}/{}/{}.{}.{}{}.nc4'.format(url, month_days[0:2], collection_number, collection_longname, year, month_days)
               for month_days in pd.date_range(year + '-' + month_begin + '-' + day_begin, year + '-' + month_end + '-' + day_end, freq='D').strftime("%m%d").tolist()]


# Get the number of files
len_files_month = len(files_month)


print("{} files to be opened:".format(len_files_month))
print("files_month", files_month)

# Read dataset URLs
ds = xr.open_mfdataset(files_month)


# MERRA-2 M2T1NXLND collection (hourly) to get Total precipitation variable (PRECTOTLAND).
collection_shortname = 'M2T1NXLND'
collection_longname = 'tavg1_2d_lnd_Nx'
collection_number = 'MERRA2_400'
MERRA2_version = '5.12.4'


# OPeNDAP URL
url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/{}.{}/{}'.format(
    collection_shortname, MERRA2_version, year)
files_month = ['{}/{}/{}.{}.{}{}.nc4'.format(url, month_days[0:2], collection_number, collection_longname, year, month_days)
               for month_days in pd.date_range(year + '-' + month_begin + '-' + day_begin, year + '-' + month_end + '-' + day_end, freq='D').strftime("%m%d").tolist()]

# Get the number of files
len_files_month = len(files_month)


print("{} files to be opened:".format(len_files_month))
print("files_month", files_month)

# Read dataset URLs
ds_precip = xr.open_mfdataset(files_month)


# MERRA-2 M2T1NXRAD collection (hourly) to get Cloud Fraction variable (CLDTOT).
collection_shortname = 'M2T1NXRAD'
collection_longname = 'tavg1_2d_rad_Nx'
collection_number = 'MERRA2_400'
MERRA2_version = '5.12.4'


# OPeNDAP URL
url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/{}.{}/{}'.format(
    collection_shortname, MERRA2_version, year)
files_month = ['{}/{}/{}.{}.{}{}.nc4'.format(url, month_days[0:2], collection_number, collection_longname, year, month_days)
               for month_days in pd.date_range(year + '-' + month_begin + '-' + day_begin, year + '-' + month_end + '-' + day_end, freq='D').strftime("%m%d").tolist()]

# Get the number of files
len_files_month = len(files_month)


print("{} files to be opened:".format(len_files_month))
print("files_month", files_month)

# Read dataset URLs
ds_cloud = xr.open_mfdataset(files_month)


# extract values from all datasets based on location (Alabama plant)
alabama_plant_temp_wind = ds.sel(lat=31.48, lon=-87.91, method='nearest')
alabama_plant_temp_wind = alabama_plant_temp_wind[['T2M', 'V2M']]

alabama_plant_precip = ds_precip.sel(lat=31.48, lon=-87.91, method='nearest')
alabama_plant_precip = alabama_plant_precip[['PRECTOTLAND']]

alabama_plant_cloud = ds_cloud.sel(lat=31.48, lon=-87.91, method='nearest')
alabama_plant_cloud = alabama_plant_cloud[['CLDTOT']]


# Resample all datasets by day
alabama_plant_temp_wind_mean = alabama_plant_temp_wind.resample(
    time="1D").mean(dim='time', skipna=True)
alabama_plant_precip_mean = alabama_plant_precip.resample(
    time="1D").mean(dim='time', skipna=True)
alabama_plant_cloud_mean = alabama_plant_cloud.resample(
    time="1D").mean(dim='time', skipna=True)

# Convert datasets to pandas df and save to CSV file.
alabama_plant_temp_wind_mean_df = alabama_plant_temp_wind_mean.to_dataframe()
alabama_plant_precip_mean_df = alabama_plant_precip_mean.to_dataframe()
alabama_plant_cloud_mean_df = alabama_plant_cloud_mean.to_dataframe()

merged_dfs = alabama_plant_temp_wind_mean_df.merge(
    alabama_plant_precip_mean_df, on='time').merge(alabama_plant_cloud_mean_df, on='time')
merged_dfs = merged_dfs[['T2M', 'V2M', 'PRECTOTLAND', 'CLDTOT']]

merged_dfs.to_csv(home_dir + '/merra2_daily_mean_January_2019.csv')
