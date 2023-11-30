"""BarSim"""

# MIT License

# Copyright (c) 2023 Guillaume Rongier, Joep Storms, Andrea Cuesta Cano

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright notice: Technische Universiteit Delft hereby disclaims all copyright
# interest in the program pyBarSim written by the Author(s).
# Prof. Dr. Ir. J.D. Jansen, Dean of the Faculty of Civil Engineering and Geosciences


import math
import random
import numpy as np
from scipy import interpolate
import numba as nb
import xarray as xr
import pyvista as pv


################################################################################
# BarSim 2D

@nb.jit(nopython=True)
def _initialize_fluxes(local_factor_shoreface, local_factor_backbarrier, sediment_size):
    """
    Adopted from Guillen and Hoekstra 1996, 1997, but slightly different (this
    is only for the >0.125mm fraction)
    """
    flux_shoreface_basic = np.zeros(len(sediment_size))
    flux_overwash_basic = np.zeros(len(sediment_size))
    for j in range (len(sediment_size)):
        flux_shoreface_basic[j] = local_factor_shoreface * (110 + 590 * (0.125/(sediment_size[j]*0.001))**2.5)
        flux_overwash_basic[j] = local_factor_backbarrier * (110 + 590 * (0.125/(sediment_size[j]*0.001))**2.5)
        
    return flux_shoreface_basic, flux_overwash_basic


@nb.jit(nopython=True)
def _correct_travel_distance(allow_storms, is_storm, flux_shoreface_basic, flux_overwash_basic,
                             max_wave_height, n_grain_sizes, correction_factor=0.25):
    """
    Corrects travel distance for storm sediment transport processes (e.g. downwelling).
    """
    flux_shoreface = np.zeros(n_grain_sizes)
    flux_overwash = np.zeros(n_grain_sizes)
    if allow_storms == False: # no events
        for i in range (n_grain_sizes):
            flux_shoreface[i] = flux_shoreface_basic[i]
            flux_overwash[i] = flux_overwash_basic[i]
    else: # events
        if is_storm == True:
            correction_shoreface = correction_factor*max_wave_height
            correction_backbarrier = correction_factor*max_wave_height
            for i in range (n_grain_sizes):
                flux_shoreface[i] = flux_shoreface_basic[i]*correction_shoreface
                flux_overwash[i] = flux_overwash_basic[i]*correction_backbarrier
        else:
            for i in range (n_grain_sizes):
                flux_shoreface[i] = flux_shoreface_basic[i]
                flux_overwash[i] = flux_overwash_basic[i]
                
    return flux_shoreface, flux_overwash


@nb.jit(nopython=True)
def _update_fluxes(allow_storms, is_storm, time, max_wave_height_storm, max_wave_height_fair_weather,
                   dt_fair_weather, dt_storm, flux_shoreface_basic, flux_overwash_basic, n_grain_sizes):
    """
    Updates fluxes when storms are allowed.
    """
    if allow_storms == False: # no events
        max_wave_height = max_wave_height_fair_weather
        T = 5. # Why this fixed value?
        dt = dt_fair_weather
        time += dt
    else: # events
        if is_storm == True:
            max_wave_height = max_wave_height_storm + 2.*random.random()
            T = 4. + 0.5*random.random()
            # wavebase_depth = T * 0.5
            dt = dt_storm
            time += dt
        else:
            max_wave_height = max_wave_height_fair_weather + 2.*random.random()
            T = 2.5 + 0.5*random.random()
            # wavebase_depth = T * 0.5
            dt = dt_fair_weather
            time += dt

    flux_shoreface, flux_overwash = _correct_travel_distance(allow_storms,
                                                             is_storm,
                                                             flux_shoreface_basic,
                                                             flux_overwash_basic,
                                                             max_wave_height,
                                                             n_grain_sizes)
    
    return max_wave_height, T, dt, time, flux_shoreface, flux_overwash


@nb.jit(nopython=True)
def _compute_shields_parameter(sediment_size):
    """
    Computes the dimensionless threshold orbital velocity, a Shields-type parameter
    defined by le Roux (2001, https://doi.org/10.1016/S0037-0738(01)00105-1).
    """
    Dd = np.zeros(len(sediment_size))
    for j in range(len(sediment_size)):
        Dd[j] = (sediment_size[j]/1000) * (1 * 981 * 2 / 0.01**2)**(1/3)    
    Wd = np.zeros(len(sediment_size))
    for j in range(len(sediment_size)):
        # TODO: Need some "=<" or ">="
        if Dd[j] < 1.2538:
            Wd[j] = 0.2354 * Dd[j]**2
        if 1.2538 < Dd[j] < 2.9074:
            Wd[j] = (0.208*Dd[j] - 0.0652)**(3/2)
        if 2.9074 < Dd[j] < 22.9866:
            Wd[j] = 0.2636*Dd[j] - 0.37
        if 22.9866 < Dd[j] < 134.9215:
            Wd[j] = (0.8255*Dd[j] - 5.4)**(2/3)
        if 134.9215 < Dd[j] < 1750.:
            Wd[j] = (2.531*Dd[j] + 160)**0.5
    dimless_thres_orbital_velocity = np.zeros(len(sediment_size))
    for j in range(len(sediment_size)):
        if Wd[j] > 0:
            dimless_thres_orbital_velocity[j] = 0.0246 * Wd[j]**(-0.55)
                         
    return dimless_thres_orbital_velocity


@nb.jit(nopython=True)
def _compute_orbital_velocity(elevation, sea_level, max_wave_height, T, dimless_thres_orbital_velocity,
                              sediment_size, i_coastline):
    """
    Computes the actual horizontal orbital velocity (m/s) based on Komar, Beach
    processes, 2nd edition, p163 and 164 and the max. horizontal orbital velocity
    (m/s) for each grain size class based on Le Roux (2001, eq. 39, unit = cm/s).
    """
    L_int = np.zeros(len(elevation))
    L_deep = (9.81 * T**2) / (2*3.1428)  # calculate deep water wave length
    for i in range(i_coastline + 1, len(elevation)):
        tmp_double = 2*3.1428*(sea_level - elevation[i])/L_deep
        L_int[i] = L_deep*math.sqrt(math.tanh(tmp_double))
    orbital_velocity = np.zeros(len(elevation))
    for i in range(i_coastline + 1, len(elevation)):
        if L_int[i] != 0.:
            num = T * math.sinh(2 * 3.1428 * (sea_level - elevation[i]) / L_int[i])
            if num != 0.:
                orbital_velocity[i] = 3.1428*max_wave_height/num
    orbital_velocity[i_coastline] = 20. # what is this number?

    orbital_velocity_max = np.zeros(len(sediment_size) + 1)
    for i in range(len(sediment_size)):
        orbital_velocity_max[i] = (-0.01 * ((dimless_thres_orbital_velocity[i] * 981 * sediment_size[i] * 0.0001 * 2) ** 2 / (1 * 0.01 / T)) + 1.3416 * ((dimless_thres_orbital_velocity[i] * 981 * sediment_size[i] * 0.0001 * 2) / ((1 * 0.01) / T) ** 0.5) - 0.6485) / 100.
        
    return orbital_velocity, orbital_velocity_max


@nb.jit(nopython=True)
def _decompose_domain(elevation, sea_level, max_wave_height, T, sediment_size,
                      dimless_thres_orbital_velocity):
    """
    Decomposes the domain by returning the indices of the mainland, the backbarrier,
    the coastline, and the wave base.
    TODO: Check that this is right
    """    
    i_coastline = len(elevation) - 1
    while elevation[i_coastline] <= sea_level and i_coastline > 0:
        i_coastline -= 1 # First dry cel, checked and it is ok

    i_backbarrier = i_coastline - 1
    while elevation[i_backbarrier] > sea_level and i_backbarrier >= 0:
        i_backbarrier -= 1
    if i_backbarrier == -1:
        i_backbarrier = i_coastline
    else:
        i_backbarrier += 1 # i_backbarrier is the final dry cell landward of i_coastline

    i_mainland = 0
    while elevation[i_mainland] > sea_level and i_mainland < len(elevation):
        i_mainland += 1
    i_mainland -= 1
            
    i_wavebase = len(elevation) - 1 # Wint to Wdeep intersection (may be too deep)
    orbital_velocity, orbital_velocity_max = _compute_orbital_velocity(elevation, sea_level,
                                                                       max_wave_height, T,
                                                                       dimless_thres_orbital_velocity,
                                                                       sediment_size, i_coastline)
    while orbital_velocity[i_wavebase] <= orbital_velocity_max[0] and i_wavebase > 0:  #for the finest fraction
        i_wavebase -= 1
            
    return i_mainland, i_backbarrier, i_coastline, i_wavebase


@nb.jit(nopython=True)
def _classify_facies(is_storm, i_mainland, i_backbarrier, i_coastline, i_wavebase,
                     i_wavebase_event, i_wavebase_fw, n_x):
    """
    Classifies the domain into facies.
    """
    if is_storm == False:
        i_wavebase_fw = i_wavebase
    else:
        i_wavebase_event = i_wavebase

    # TODO: Check with Joep that's ok to turn those into int
    facies = np.empty(n_x, np.int8)
    for i in range(0, i_mainland):
        # Coastal plain
        facies[i] = 2
    for i in range(i_mainland, i_backbarrier):
        # Lagoon
        facies[i] = 3
    for i in range(i_backbarrier, i_coastline):
        # Barrier island
        facies[i] = 4
    for i in range(i_coastline, i_wavebase):
        # Upper shoreface
        facies[i] = 5
    for i in range(i_wavebase_fw, i_wavebase_event):
        # Lower shoreface
        facies[i] = 6
    for i in range(i_wavebase_event, len(facies)):
        # Offshore
        facies[i] = 7
        
    return facies, i_wavebase_event, i_wavebase_fw


@nb.jit(nopython=True)
def _erode_stratigraphy(stratigraphy, erosion_total, i_time, i_coastline, i_wavebase):
    """
    Erodes the stratigraphic record along the shoreface and shelf.
    """
    n_grain_sizes = stratigraphy.shape[0]
    erosion = np.zeros((n_grain_sizes, len(erosion_total)))
    for i in range(i_coastline, i_wavebase + 1):

        layer_thickness = 0.
        for j in range(n_grain_sizes):
            layer_thickness += stratigraphy[j, i_time - 1, i]

        if erosion_total[i] > layer_thickness:
            
            erosion_sum = 0.
            sum_strat = np.zeros(n_grain_sizes)
            t = 0
            while erosion_total[i] > erosion_sum:
                t += 1
                for j in range(n_grain_sizes):
                    erosion_sum += stratigraphy[j, i_time - t, i]
                    sum_strat[j] += stratigraphy[j, i_time - t, i]

            last_layer_thickness = 0.
            for j in range(n_grain_sizes):
                last_layer_thickness += stratigraphy[j, i_time - t, i]
            
            fraction_left = (erosion_sum - erosion_total[i])/last_layer_thickness
            for l in range(1, t): # CHECK THIS LINE tmp OR tmp-1???
                for j in range(n_grain_sizes):
                    stratigraphy[j, i_time - l, i] = 0.
            for j in range(n_grain_sizes):
                stratigraphy[j, i_time - t, i] *= fraction_left
                erosion[j, i] = sum_strat[j] - stratigraphy[j, i_time - t, i]

        else:
            # TODO: What if layer_thickness is equal to 0?
            fraction_left = erosion_total[i]/layer_thickness
            for j in range(n_grain_sizes):
                erosion[j, i] = fraction_left*stratigraphy[j, i_time - 1, i]
                stratigraphy[j, i_time - 1, i] *= (1 - fraction_left)

    return stratigraphy, erosion


@nb.jit(nopython=True)
def _erode(elevation, stratigraphy, sea_level, erodibility, max_wave_height,
           max_wave_height_fair_weather, i_time, i_coastline, i_wavebase):
    """
    Erodes the domain based on Storms (2003) without reflection correction cd(t).
    """
    erosion_total = np.zeros(len(elevation[i_time]))
    for i in range(i_coastline, i_wavebase + 1):
        erosion_total[i] = erodibility * (max_wave_height/max_wave_height_fair_weather) * ((elevation[i_time, i] - elevation[i_time, i_wavebase]) / (sea_level - elevation[i_time, i_wavebase]))**3
        
    stratigraphy, erosion = _erode_stratigraphy(stratigraphy, erosion_total, 
                                                i_time, i_coastline, i_wavebase)
        
    return stratigraphy, erosion


@nb.jit(nopython=True)
def _distribute_fluxes(erosion, sediment_supply, sea_level, sea_level_prev, washover_fraction,
                       max_width_backbarrier, sediment_fraction, i_mainland, i_backbarrier,
                       i_coastline, i_wavebase, dt, spacing):
    """
    Distributes the fluxes along the domain.
    """
    n_grain_sizes = len(sediment_fraction)
    total_flux = np.zeros(n_grain_sizes)
    for i in range(i_coastline, i_wavebase + 1):
        for j in range(n_grain_sizes): 
            total_flux[j] += erosion[j, i]*spacing
    for j in range(n_grain_sizes):
        total_flux[j] += sediment_supply*sediment_fraction[j]*dt

    total_flux_washover = np.zeros(n_grain_sizes)
    total_flux_shoreface = np.zeros(n_grain_sizes)
    if (i_backbarrier > 1
        and i_coastline - i_mainland > 0
        and (i_coastline - i_backbarrier)*spacing < max_width_backbarrier
        and sea_level >= sea_level_prev
        and washover_fraction > 0.): 
        for j in range(2, n_grain_sizes):    #NOTE!! ASSUME GRAINSIZE 3 and 4 ARE SAND
            total_flux_washover[j] = washover_fraction*total_flux[j]
        for j in range(n_grain_sizes):
            total_flux_shoreface[j] = total_flux[j] - total_flux_washover[j]
            
    return total_flux, total_flux_washover, total_flux_shoreface


@nb.jit(nopython=True)
def _deposit_washover(elevation, total_flux_washover, flux_overwash, sea_level,
                      sea_level_prev, washover_fraction, max_width_backbarrier,
                      depth_factor_backbarrier, max_barrier_height_backbarrier,
                      i_time, i_mainland, i_backbarrier, i_coastline, spacing):
    """
    Deposits washover sediments based on the restriction algorithm (travel distance ~ expected height).
    """
    n_grain_sizes = len(total_flux_washover)
    sediment_flux = np.zeros((n_grain_sizes, len(elevation[i_time])))
    deposition_washover = np.zeros((n_grain_sizes, len(elevation[i_time])))
    if (i_backbarrier > 1
        and i_coastline - i_mainland > 0
        and (i_coastline - i_backbarrier)*spacing < max_width_backbarrier
        and sea_level >= sea_level_prev
        and washover_fraction > 0.):
        for j in range(2, n_grain_sizes):
            sediment_flux[j, i_coastline - 1] = total_flux_washover[j]

        for i in range(i_coastline - 1, i_mainland, -1):
            H_norm = (elevation[i_time, i] - sea_level)/max_barrier_height_backbarrier

            f_add = np.zeros(n_grain_sizes)
            for j in range(2, n_grain_sizes):
                f_add[j] = flux_overwash[j]*(1. + 2.71828283**(H_norm*depth_factor_backbarrier))
                sediment_flux[j, i - 1] = sediment_flux[j, i] - sediment_flux[j, i]*spacing/f_add[j]  
                deposition_washover[j, i] = (sediment_flux[j, i] - sediment_flux[j, i - 1])/spacing
                
    return sediment_flux, deposition_washover


@nb.jit(nopython=True)
def _redistribute_fluxes(sediment_flux, total_flux, total_flux_shoreface, sea_level,
                         sea_level_prev, washover_fraction, max_width_backbarrier,
                         i_backbarrier, i_coastline, i_mainland, spacing):
    """
    Redistributes fluxes in the domain.
    """
    n_grain_sizes = sediment_flux.shape[0]
    if (i_backbarrier > 1
        and i_coastline - i_mainland > 0
        and (i_coastline - i_backbarrier)*spacing < max_width_backbarrier
        and sea_level >= sea_level_prev
        and washover_fraction > 0.):
        for j in range(n_grain_sizes):
            sediment_flux[j, i_coastline] = sediment_flux[j, i_mainland] + total_flux_shoreface[j]
    else:
        for j in range(n_grain_sizes):
            sediment_flux[j, i_coastline] = total_flux[j]
        
    return sediment_flux


@nb.jit(nopython=True)
def _deposit_tidal(elevation, deposition_washover, sediment_flux, sea_level, tidal_amplitude,
                   min_tidal_area_for_transport, tide_sand_fraction, fallout_rate_backbarrier,
                   is_storm, i_time, i_backbarrier, i_mainland, i_coastline, dt, spacing):
    """
    Deposits tidal sediments.
    """
    n_grain_sizes = sediment_flux.shape[0]
    deposition_tidal = np.zeros((n_grain_sizes, len(elevation[i_time])))
    deposition_tidal_tot = np.zeros(n_grain_sizes)
    
    sediment_flux_total = 0.
    for j in range(n_grain_sizes):
        sediment_flux_total += sediment_flux[j, i_coastline]
    
    if is_storm == False and tidal_amplitude > 0:
        
        deposition_tidal_sum = 0.
        backbarrier_accommodation = 0
        # TODO: should that be changed for not taking into account if sea_level - elevation[i_time, i] <= 0?
        len_backbarrier = 0
        # Determine backbarrier accommodation
        for i in range(i_backbarrier - 1, i_mainland, -1):
            if sea_level - elevation[i_time, i] > 0.:
                backbarrier_accommodation += (sea_level - elevation[i_time, i])*spacing
                len_backbarrier += 1
        # Minimum tidal basin size: USE ASMITA HERE? Area that is wet is 
        # proportional to the Tidal Amplitude. Also assume minimal hydraulic
        # gradient befor tidal processes become active
        if backbarrier_accommodation < min_tidal_area_for_transport - tidal_amplitude*spacing*len_backbarrier:
            tidal_supply = False
        else:
            tidal_supply = True
            
        # Max deposition rate based on Tidal amplitude, Backbarrier width and
        # time step, limited by total sediment flux
        dh_ratio_acc = 1. # We need a default because the if statement below doesn't have an else, but is this a good default?
        if tidal_supply == True and tidal_amplitude > 0:
            tidal_deposition_capacity = len_backbarrier*spacing*tidal_amplitude*dt*0.001 # 1 is tuning parameter

            if (tidal_deposition_capacity > sediment_flux_total):
                tidal_deposition_capacity = sediment_flux_total

            if (tidal_deposition_capacity > backbarrier_accommodation):
                dh_ratio_acc = 1.
            else:
                dh_ratio_acc = tidal_deposition_capacity/backbarrier_accommodation

        # Max deposition rate cannot exceed the total eroded sediment from the shoreface
        for j in range(n_grain_sizes):
            deposition_tidal_tot[j] = dh_ratio_acc*sediment_flux[j, i_coastline]/spacing
            deposition_tidal_sum += deposition_tidal_tot[j]

        if (deposition_tidal_sum <= 0):
            deposition_tidal_sum = 0.0000001 # Prevent dividing by zero negative sediment supply

        # Deposition finest fraction
        for i in range(i_backbarrier - 1, i_mainland, -1):
            if sea_level - elevation[i_time, i] > 0.:
                for j in range(n_grain_sizes - 2):
                    deposition_tidal[j, i] = dh_ratio_acc*(sea_level - elevation[i_time, i])*deposition_tidal_tot[j]/deposition_tidal_sum # grain size distribution tidal deposits (sum should be 1)
                # Coarsest fraction less
                deposition_tidal[n_grain_sizes - 1, i] = tide_sand_fraction*dh_ratio_acc*(sea_level - elevation[i_time, i])*deposition_tidal_tot[n_grain_sizes - 1]/deposition_tidal_sum # grain size distribution tidal deposits (sum should be 1)
            
            # Leftover sediment for SHOREFACE deposition
            for j in range(n_grain_sizes):
                sediment_flux[j, i_coastline] -= deposition_tidal[j, i]*spacing

    # Update layer thickness (dh) and cellheight and assign remaining sediment for shoreface sedimentation
    if (fallout_rate_backbarrier > 0):
        for i in range(i_backbarrier - 1, i_mainland, -1): # fallout rate in backbarrier (organics + fines)
            if sea_level - elevation[i_time, i] > 0.:
                if fallout_rate_backbarrier*dt < (sea_level - elevation[i_time, i]): # in case of accommodation
                    deposition_washover[0, i] += fallout_rate_backbarrier*dt
                else:
                    deposition_washover[0, i] += (sea_level - elevation[i_time, i]) # no BB deposition above SL
            
    return sediment_flux, deposition_washover, deposition_tidal


@nb.jit(nopython=True)
def _deposit_shoreface(elevation, sediment_flux, flux_shoreface, sea_level,
                       max_barrier_height_shoreface, fallout_rate_shoreface,
                       depth_factor_shoreface, i_time, i_coastline, spacing, dt):
    """
    Deposits shoreface sediments.
    """
    n_grain_sizes = sediment_flux.shape[0]
    deposition_shoreface = np.zeros((n_grain_sizes, len(elevation[i_time])))
    # TODO: Would it be possible to do sediment_flux[j, i - 1] - sediment_flux[j, i]?
    for i in range(i_coastline, len(elevation[i_time]) - 1):
        H_norm = (elevation[i_time, i] - sea_level)/max_barrier_height_shoreface - max_barrier_height_shoreface
        f_add = np.zeros(n_grain_sizes)
        for j in range(n_grain_sizes):
            f_add[j] = flux_shoreface[j]*(1 + 2.71828283**(H_norm*depth_factor_shoreface))
        for j in range(n_grain_sizes):
            sediment_flux[j, i + 1] = sediment_flux[j, i] - sediment_flux[j, i]*spacing/f_add[j]
        for j in range(n_grain_sizes):
            deposition_shoreface[j, i] = (sediment_flux[j, i] - sediment_flux[j, i + 1])/spacing
        deposition_shoreface[0, i] += fallout_rate_shoreface*dt
    
    return sediment_flux, deposition_shoreface


@nb.jit(nopython=True)
def _update_elevation(elevation, erosion, deposition_washover, deposition_tidal,
                      deposition_shoreface, i_time, i_mainland):
    """
    Updates the topography following erosion and deposition.
    """
    for i in range(i_mainland, elevation.shape[1]):    
        for j in range(erosion.shape[0]):
            elevation[i_time, i] += deposition_washover[j, i] + deposition_tidal[j, i] + deposition_shoreface[j, i] - erosion[j, i]
            
    return elevation


@nb.jit(nopython=True)
def _deposit_stratigraphy(stratigraphy, deposition_washover, deposition_tidal,
                          deposition_shoreface, i_time, i_mainland):
    """
    Deposits the stratigraphic record.
    """
    for i in range(i_mainland, stratigraphy.shape[2]):    
        for j in range(stratigraphy.shape[0]):
            stratigraphy[j, i_time, i] = deposition_washover[j, i] + deposition_tidal[j, i] + deposition_shoreface[j, i]
            
    return stratigraphy


@nb.jit(nopython=True)
def _deposit(elevation, stratigraphy, erosion, sediment_supply, flux_overwash,
             flux_shoreface, sea_level, washover_fraction, max_width_backbarrier,
             depth_factor_backbarrier, max_barrier_height_backbarrier, tidal_amplitude,
             min_tidal_area_for_transport, tide_sand_fraction, fallout_rate_backbarrier,
             max_barrier_height_shoreface, fallout_rate_shoreface, depth_factor_shoreface,
             is_storm, sediment_fraction, i_time, i_mainland, i_backbarrier, i_coastline,
             i_wavebase, dt, spacing):
    """
    Deposits sediments in the domain.
    """
    total_flux, total_flux_washover, total_flux_shoreface = _distribute_fluxes(erosion, sediment_supply[i_time],
                                                                               sea_level[i_time], sea_level[i_time - 1],
                                                                               washover_fraction, max_width_backbarrier,
                                                                               sediment_fraction, i_mainland, i_backbarrier,
                                                                               i_coastline, i_wavebase, dt, spacing)
    sediment_flux, deposition_washover = _deposit_washover(elevation, total_flux_washover, flux_overwash, sea_level[i_time],
                                                           sea_level[i_time - 1], washover_fraction, max_width_backbarrier,
                                                           depth_factor_backbarrier, max_barrier_height_backbarrier, i_time,
                                                           i_mainland, i_backbarrier, i_coastline, spacing)
    sediment_flux = _redistribute_fluxes(sediment_flux, total_flux, total_flux_shoreface, sea_level[i_time], sea_level[i_time - 1],
                                         washover_fraction, max_width_backbarrier, i_backbarrier, i_coastline, i_mainland, spacing)
    sediment_flux, deposition_washover, deposition_tidal = _deposit_tidal(elevation, deposition_washover, sediment_flux,
                                                                          sea_level[i_time], tidal_amplitude, min_tidal_area_for_transport,
                                                                          tide_sand_fraction, fallout_rate_backbarrier, is_storm, i_time,
                                                                          i_backbarrier, i_mainland, i_coastline, dt, spacing)
    sediment_flux, deposition_shoreface = _deposit_shoreface(elevation, sediment_flux, flux_shoreface, sea_level[i_time],
                                                             max_barrier_height_shoreface, fallout_rate_shoreface,
                                                             depth_factor_shoreface, i_time, i_coastline, spacing, dt)
    
    elevation = _update_elevation(elevation, erosion, deposition_washover, deposition_tidal,
                                  deposition_shoreface, i_time, i_mainland)
    stratigraphy = _deposit_stratigraphy(stratigraphy, deposition_washover, deposition_tidal,
                                         deposition_shoreface, i_time, i_mainland)
    
    return elevation, stratigraphy
    

@nb.jit(nopython=True)
def _run(initial_elevation, sea_level_curve, sediment_supply_curve, spacing,
         run_time, dt_fair_weather, dt_storm, max_wave_height_fair_weather,
         allow_storms, start_with_storm, max_wave_height_storm, tidal_amplitude,
         min_tidal_area_for_transport, sediment_size, sediment_fraction,
         initial_substratum, erodibility, washover_fraction, tide_sand_fraction,
         depth_factor_backbarrier, depth_factor_shoreface, local_factor_shoreface,
         local_factor_backbarrier, fallout_rate_backbarrier, fallout_rate_shoreface,
         max_width_backbarrier, max_barrier_height_shoreface,
         max_barrier_height_backbarrier, seed):
    """
    Runs BarSim for a given run time.
    """
    random.seed(seed)

    if allow_storms == True:
        n_time_steps = math.ceil(2*run_time/(dt_fair_weather + dt_storm)) + 1
    else:
        n_time_steps = math.ceil(run_time/dt_fair_weather) + 1
    
    time = np.zeros(n_time_steps)
    sea_level = np.zeros(n_time_steps)
    sea_level[0] = sea_level_curve[0, 1]
    sediment_supply = np.zeros(n_time_steps)
    sediment_supply[0] = sediment_supply_curve[0, 1]
    elevation = np.zeros((n_time_steps, len(initial_elevation)))
    elevation[0] = initial_elevation    
    stratigraphy = np.zeros((len(sediment_size), n_time_steps, len(initial_elevation)))
    stratigraphy[:, 0] = initial_substratum
    facies = np.ones((n_time_steps, len(initial_elevation)), np.int8)

    flux_shoreface_basic, flux_overwash_basic = _initialize_fluxes(local_factor_shoreface,
                                                                   local_factor_backbarrier,
                                                                   sediment_size)
    dimless_thres_orbital_velocity = _compute_shields_parameter(sediment_size)
    i_wavebase_event = 0
    i_wavebase_fw = 0

    is_storm = start_with_storm

    i_time = 1
    while time[i_time - 1] < run_time:
        
        max_wave_height, T, dt, time[i_time], flux_shoreface, flux_overwash = _update_fluxes(allow_storms, is_storm, time[i_time - 1], max_wave_height_storm,
                                                                                             max_wave_height_fair_weather, dt_fair_weather, dt_storm,
                                                                                             flux_shoreface_basic, flux_overwash_basic, len(sediment_size))

        sea_level[i_time] = np.interp(time[i_time],
                                      sea_level_curve[:, 0],
                                      sea_level_curve[:, 1])
        sediment_supply[i_time] = np.interp(time[i_time],
                                            sediment_supply_curve[:, 0],
                                            sediment_supply_curve[:, 1])
        elevation[i_time] = elevation[i_time - 1]
        
        i_mainland, i_backbarrier, i_coastline, i_wavebase = _decompose_domain(elevation[i_time], sea_level[i_time], max_wave_height, T,
                                                                               sediment_size, dimless_thres_orbital_velocity)
        facies[i_time], i_wavebase_event, i_wavebase_fw = _classify_facies(is_storm, i_mainland, i_backbarrier, i_coastline, i_wavebase,
                                                                           i_wavebase_event, i_wavebase_fw, len(elevation[i_time]))
        stratigraphy, erosion = _erode(elevation, stratigraphy, sea_level[i_time], erodibility, max_wave_height,
                                       max_wave_height_fair_weather, i_time, i_coastline, i_wavebase)        
        elevation, stratigraphy = _deposit(elevation, stratigraphy, erosion, sediment_supply, flux_overwash, flux_shoreface, sea_level,
                                           washover_fraction, max_width_backbarrier, depth_factor_backbarrier, max_barrier_height_backbarrier,
                                           tidal_amplitude, min_tidal_area_for_transport, tide_sand_fraction, fallout_rate_backbarrier,
                                           max_barrier_height_shoreface, fallout_rate_shoreface, depth_factor_shoreface, is_storm,
                                           sediment_fraction, i_time, i_mainland, i_backbarrier, i_coastline, i_wavebase, dt, spacing)

        i_time += 1
        if is_storm == False:
            is_storm = True
        else:
            is_storm = False
    
    return (time[:i_time], sea_level[:i_time], sediment_supply[:i_time],
            elevation[:i_time], stratigraphy[:, :i_time], facies[:i_time])


@nb.jit(nopython=True)
def _compute_mean_grain_size(stratigraphy, sediment_size):
    """
    Computes the mean grain size from the stratigraphy.
    """
    mean = np.zeros(stratigraphy.shape[1:])
    for k in range(stratigraphy.shape[1]):
        for i in range(stratigraphy.shape[2]):
            stratigraphy_total = np.sum(stratigraphy[:, k, i])
            if stratigraphy_total > 0.:
                for j in range(stratigraphy.shape[0]):
                    mean[k, i] += sediment_size[j]*stratigraphy[j, k, i]
                mean[k, i] /= stratigraphy_total
                
    return mean


@nb.jit(nopython=True)
def _compute_std_grain_size(stratigraphy, grain_size, to_phi):
    """
    Computes the sorting term from the stratigraphy.
    """
    if to_phi == True:
        grain_size = -np.log2(grain_size/1000.)

    std = np.zeros(stratigraphy.shape[1:])
    for k in range(stratigraphy.shape[1]):
        for i in range(stratigraphy.shape[2]):
            stratigraphy_total = np.sum(stratigraphy[:, k, i])
            if stratigraphy_total > 0.:
                mean = 0.
                for j in range(stratigraphy.shape[0]):
                    mean += stratigraphy[j, k, i]*grain_size[j]
                mean /= stratigraphy_total
                for j in range(stratigraphy.shape[0]):
                    std[k, i] += stratigraphy[j, k, i]*(grain_size[j] - mean)**2
                std[k, i] = math.sqrt(std[k, i]/stratigraphy_total)
                
    return std


@nb.jit(nopython=True)
def _regrid_stratigraphy(elevation, time_stratigraphy, time_facies, z_min, z_max, dz):
    """
    Interpolates the time stratigraphy on a regular grid in space.
    """
    z_corner = np.linspace(z_min, z_max, int((z_max - z_min)/dz) + 1)
    stratigraphy = np.zeros((time_stratigraphy.shape[0],
                             len(z_corner) - 1,
                             time_stratigraphy.shape[2]))
    facies = np.zeros((8, len(z_corner) - 1, time_facies.shape[1]))
    
    for i in range(time_stratigraphy.shape[2]):
        k = len(z_corner) - 2
        l = time_stratigraphy.shape[1] - 1
        layer_top = elevation[i]
        layer_bottom = layer_top - np.sum(time_stratigraphy[:, l, i])
        while k >= 0 and l >= 0:
            if (layer_top > layer_bottom
                and layer_bottom < z_corner[k + 1]
                and z_corner[k] < layer_top):
                ratio = (min(z_corner[k + 1], layer_top) - max(z_corner[k], layer_bottom))/(layer_top - layer_bottom)
                stratigraphy[:, k, i] += ratio*time_stratigraphy[:, l, i]
                ratio = (min(z_corner[k + 1], layer_top) - max(z_corner[k], layer_bottom))/dz
                facies[time_facies[l, i], k, i] += ratio
            if z_corner[k + 1] > elevation[i]:
                facies[0, k, i] = (z_corner[k + 1] - max(elevation[i], z_corner[k]))/dz
            if l == 0 and z_corner[k] < layer_bottom:
                facies[0, k, i] = (min(z_corner[k + 1], layer_bottom) - z_corner[k])/dz
            if l > 0 and z_corner[k] < layer_bottom:
                l -= 1
                layer_top = layer_bottom
                layer_bottom = layer_top - np.sum(time_stratigraphy[:, l, i])
            else:
                k -= 1

    return stratigraphy, facies


class BarSim2D:
    """
    2D implementation of the multiple grain-size, process-response model BarSim.

    Parameters
    ----------
    initial_elevation : array-like of shape (n_x,)
        The initial elevation (m).
    sea_level_curve : array-like of shape (n_t, 2)
        The inflection points of sea level in time (yr, m).
    sediment_supply_curve : array-like of shape (n_t, 2)
        The inflection points of sediment supply in time (yr, m2/yr).
    spacing : float, default=100.
        The grid spatial resolution (m).
    max_wave_height_fair_weather : float, default=1.5
        Maximum fair-weather wave height used to simulate a time-series of wave
        height under fair-weather conditions (m).
    allow_storms : bool, default=True
        If True, allows storm events, otherwise only fair weather is simulated.
    start_with_storm : bool, default=False
        If True, starts the simulation with a storm, otherwise starts with fair-
        weather conditions.
    max_wave_height_storm : float, default=6.
        Maximum wave height for storm events (m).
    tidal_amplitude : float, default=2.
        Tidal amplitude (m).
    min_tidal_area_for_transport : float, default=100.
        Minimum tidal prism area required for tide induced transport to the
        backbarrier/tidal basin (m).
    sediment_size : array-like of shape (n_grain_sizes,), default=(5, 50, 125, 250)
        Diameter of the grains of the different classes of sediments within the
        influx of additional sediment by longshore drift (μm).
    sediment_fraction : array-like of shape (n_grain_sizes,), default=(0.25, 0.25, 0.25, 0.25)
        Fraction of the different classes of sediments within the influx of additional
        sediment by longshore drift. Each value must be between 0 and 1.
    initial_substratum : array-like of shape (n_grain_sizes, n_x) or (2,), default=(100., (0.25, 0.25, 0.25, 0.25))
        Initial thickness of the substratum for each grain size. It can be directly
        an array representing the substratum, or a tuple (thickness, tuple with the
        proportion of each grain size), which is converted into an array during
        initialization. Make sure that the substrate is thick enough to not be
        fully eroded during simulation.
    erodibility : float, default=0.1
        Multiplier or tuning parameter (c_e in equation 3 of Storms (2003)) to
        adjust the erosion capacity.
    washover_fraction : float, default=0.5
        Fraction of sediments going into washover instead of shoreface. It must
        be between 0 and 1.
    tide_sand_fraction : float, default=0.3
        Fraction of the sand-prone grain-size class available for tide-induced
        transport. This rule ensures that tide-induced transport is focused on
        the fine-grained fractions by limiting the sand-sized grain-size
        fraction. NOTE that the correct grain-size class is limited! It must be
        between 0 and 1.
    depth_factor_backbarrier : float, default=5.
        Parameter `A` in equation 9 (Storms et al., 2002) applied to backbarrier
        deposition.
    depth_factor_shoreface : float, default=10.
        Parameter `A` in equation 9 (Storms et al., 2002) applied to backbarrier
        deposition.
    local_factor_shoreface : float, default=1.5
        Multiplier or tuning parameter (c_g in equations 13 and 14 of Storms (2003))
        to adjust for local conditions of the sediment travel distance at the
        shoreface.
    local_factor_backbarrier : float, default=1.
        Multiplier or tuning parameter (c_g in equations 13 and 14 of Storms (2003))
        to adjust for local conditions of the sediment travel distance at the
        backbarrier.
    fallout_rate_backbarrier : float, default=0.
        Constant fall-out rate of fines (the finest sediment fraction only) in the
        backbarrier area (mm/yr). Is only active during fair-weather conditions.
    fallout_rate_shoreface : float, default=0.0002
        Constant fall-out rate of fines (the finest sediment fraction only) in the
        shoreface-shelf (mm/yr), representing pelagic sedimentation. It is only
        active during fair-weather conditions.
    max_width_backbarrier : float, default=500.
        Maximum width of the backbarrier (m).
    max_barrier_height_shoreface : float, default=1.
        Maximum height of barrier island on the seaward side (m). Higher elevations
        above sea level limit the deposition. The parameter refers to z_l in
        equation 9 in Storms et al. (2002). This is a sensitive parameter, better
        keep the default value.
    max_barrier_height_backbarrier : float, default=1.
        Maximum height of barrier island on the backbarrier side (m). Higher
        elevations above sea level limit the deposition. The parameter refers to
        z_l in equation 9 in Storms et al. (2002). This is a sensitive parameter,
        better keep the default value.
    preinterpolate_curves: bool, default=False
        If True, pre-interpolates the sea level and sediment supply curves.
    monotonic: bool, default=True
        If True, uses a monotonic interpolation for sea level and sediment supply,
        otherwise uses a cubic interpolation.
    seed : int, default=42
        Controls random number generation for reproducibility.

    Attributes
    ----------
    sequence_ : xarray.Dataset
        Dataset containing the evolution of the elevation and stratigraphy through time.
    record_ : xarray.Dataset
        Dataset containing the final stratigraphy.
        
    References
    ----------
    Storms, J.E.A., Weltje, G.J., Van Dijke, J.J., Geel, C.R., Kroonenberg, S.B. (2002) https://doi.org/10.1306/052501720226
        Process-response modeling of wave-dominated coastal systems: simulating evolution and stratigraphy on geological timescales.
    Storms, J.E.A. (2003) https://doi.org/10.1016/S0025-3227(03)00144-0
        Event-based stratigraphic simulation of wave-dominated shallow-marine environments
    """
    def __init__(self,
                 initial_elevation,
                 sea_level_curve,
                 sediment_supply_curve,
                 spacing=100.,
                 max_wave_height_fair_weather=1.5,
                 allow_storms=True,
                 start_with_storm=False,
                 max_wave_height_storm=6.,
                 tidal_amplitude=2.,
                 min_tidal_area_for_transport=100.,
                 sediment_size=(5., 50., 125., 250.),
                 sediment_fraction=(0.25, 0.25, 0.25, 0.25),
                 initial_substratum=(100., (0.25, 0.25, 0.25, 0.25)),
                 erodibility=0.1,
                 washover_fraction=0.5,
                 tide_sand_fraction=0.3,
                 depth_factor_backbarrier=5.,
                 depth_factor_shoreface=10.,
                 local_factor_shoreface=1.5,
                 local_factor_backbarrier=1.,
                 fallout_rate_backbarrier=0.,
                 fallout_rate_shoreface=0.0002,
                 max_width_backbarrier=500.,
                 max_barrier_height_shoreface=1.,
                 max_barrier_height_backbarrier=1.,
                 preinterpolate_curves=False,
                 monotonic=True,
                 seed=42):
        
        self.initial_elevation = np.asarray(initial_elevation)
        self.sea_level_curve = np.asarray(sea_level_curve)
        self.sediment_supply_curve = np.asarray(sediment_supply_curve)
        self.spacing = spacing
        self.max_wave_height_fair_weather = max_wave_height_fair_weather
        self.allow_storms = allow_storms
        self.start_with_storm = start_with_storm
        self.max_wave_height_storm = max_wave_height_storm
        self.tidal_amplitude = tidal_amplitude
        self.min_tidal_area_for_transport = min_tidal_area_for_transport
        self.sediment_size = np.asarray(sediment_size)
        self.sediment_fraction = np.asarray(sediment_fraction)
        if (len(initial_substratum) == 2
            and len(initial_substratum[1]) == len(sediment_size)):
            self.initial_substratum = np.full((len(sediment_size), len(initial_elevation)),
                                              initial_substratum[0])
            self.initial_substratum *= np.asarray(initial_substratum[1])[:, np.newaxis]
        else:
            self.initial_substratum = np.asarray(initial_substratum)
        self.erodibility = erodibility
        self.washover_fraction = washover_fraction
        self.tide_sand_fraction = tide_sand_fraction
        self.depth_factor_backbarrier = depth_factor_backbarrier
        self.depth_factor_shoreface = depth_factor_shoreface
        self.local_factor_shoreface = local_factor_shoreface
        self.local_factor_backbarrier = local_factor_backbarrier
        self.fallout_rate_backbarrier = fallout_rate_backbarrier
        self.fallout_rate_shoreface = fallout_rate_shoreface
        self.max_width_backbarrier = max_width_backbarrier
        self.max_barrier_height_shoreface = max_barrier_height_shoreface
        self.max_barrier_height_backbarrier = max_barrier_height_backbarrier
        self.preinterpolate_curves = preinterpolate_curves
        self.monotonic = monotonic
        self.seed = seed
        
    def _preinterpolate_curves(self, run_time, dt_fair_weather, dt_storm):
        """
        Preinterpolate the sea level and sediment supply curves using a cubic interpolation
        before using them in Numba.
        TODO: Would be better to have a cubic interpolation in Numba.
        """
        if self.allow_storms == True:
            n_time_steps = math.ceil(2*run_time/(dt_fair_weather + dt_storm)) + 1
        else:
            n_time_steps = math.ceil(run_time/dt_fair_weather) + 1
        
        if self.monotonic == False:
            sea_level_function = interpolate.interp1d(self.sea_level_curve[:, 0],
                                                      self.sea_level_curve[:, 1],
                                                      kind='cubic')
            sediment_supply_function = interpolate.interp1d(self.sediment_supply_curve[:, 0],
                                                            self.sediment_supply_curve[:, 1],
                                                            kind='cubic')
        else:
            sea_level_function = interpolate.PchipInterpolator(self.sea_level_curve[:, 0],
                                                               self.sea_level_curve[:, 1])
            sediment_supply_function = interpolate.PchipInterpolator(self.sediment_supply_curve[:, 0],
                                                                     self.sediment_supply_curve[:, 1])
            
        sea_level_interp = np.empty((n_time_steps, 2))
        sea_level_interp[:, 0] = np.linspace(0., run_time, n_time_steps)
        sea_level_interp[:, 1] = sea_level_function(sea_level_interp[:, 0])
        sediment_supply_interp = np.empty((n_time_steps, 2))
        sediment_supply_interp[:, 0] = sea_level_interp[:, 0]
        sediment_supply_interp[:, 1] = sediment_supply_function(sediment_supply_interp[:, 0])
        
        return sea_level_interp, sediment_supply_interp
    
    def run(self, run_time=10000., dt_fair_weather=15., dt_storm=1.):
        """
        Runs BarSim for a given run_time.

        Parameters
        ----------
        run_time : float, default=10000.
            Duration of the run (yr).
        dt_fair_weather : float, default=15.
            Time step in fair weather conditions (yr).
        dt_storm : float, default=1.
            Time step during a storm (yr).

        Returns
        -------
        self : object
            Simulator with a `sequence_` attribute.
        """
        if self.preinterpolate_curves == True:
            sea_level_curve, sediment_supply_curve = self._preinterpolate_curves(run_time, dt_fair_weather, dt_storm)
        else:
            sea_level_curve, sediment_supply_curve = self.sea_level_curve, self.sediment_supply_curve
        time, sea_level, sediment_supply, elevation, stratigraphy, facies = _run(self.initial_elevation, sea_level_curve, sediment_supply_curve,
                                                                                 self.spacing, run_time, dt_fair_weather, dt_storm, self.max_wave_height_fair_weather,
                                                                                 self.allow_storms, self.start_with_storm, self.max_wave_height_storm,
                                                                                 self.tidal_amplitude, self.min_tidal_area_for_transport, self.sediment_size,
                                                                                 self.sediment_fraction, self.initial_substratum, self.erodibility,
                                                                                 self.washover_fraction, self.tide_sand_fraction, self.depth_factor_backbarrier,
                                                                                 self.depth_factor_shoreface, self.local_factor_shoreface,
                                                                                 self.local_factor_backbarrier, self.fallout_rate_backbarrier,
                                                                                 self.fallout_rate_shoreface, self.max_width_backbarrier,
                                                                                 self.max_barrier_height_shoreface, self.max_barrier_height_backbarrier,
                                                                                 self.seed)
        self.sequence_ = xr.Dataset(
            data_vars={
                'Sea level': (
                    ('Time',),
                    sea_level,
                    {
                        'units': 'meter', 'description': 'sea level through time',
                    }
                ),
                'Sediment supply': (
                    ('Time',),
                    sediment_supply,
                    {
                        'units': 'cubic meter',
                        'description': 'sediment supply through time',
                    }
                ),
                'Elevation': (
                    ('Time', 'X'),
                    elevation,
                    {
                        'units': 'meter',
                        'description': 'elevation through time',
                    }
                ),
                'Stratigraphy': (
                    ('Grain size', 'Time', 'X'),
                    stratigraphy,
                    {
                        'units': 'meter',
                        'description': 'deposit thickness through time',
                    }
                ),
                'Facies': (
                    ('Time', 'X'),
                    facies,
                    {
                        'units': '',
                        'description': 'facies through time: 1. substratum, 2. coastal plain, 3. lagoon, 4. barrier island, 5. upper shoreface, 6. lower shoreface, 7. offshore',
                    },
                ),
            },
            coords={
                'X': np.linspace(self.spacing/2.,
                                 self.spacing*(elevation.shape[1] - 0.5),
                                 elevation.shape[1]),
                'Time': time,
                'Grain size': self.sediment_size,
            },
        )
        self.sequence_['X'].attrs['units'] = 'meter'
        self.sequence_['Time'].attrs['units'] = 'year'
        self.sequence_['Grain size'].attrs['units'] = 'micrometer'

        return self
        
    def regrid(self, z_min, z_max, dz):
        """
        Interpolates the time stratigraphy into a regular grid.

        Parameters
        ----------
        z_min : float
            Lower border of the grid (m).
        z_max : float
            Upper border of the grid (m).
        dz : float
            Resolution of the grid (m).

        Returns
        -------
        self : object
            Simulator with a `record_` attribute.
        """
        stratigraphy, facies = _regrid_stratigraphy(self.sequence_['Elevation'][-1].to_numpy(),
                                                    self.sequence_['Stratigraphy'].to_numpy(),
                                                    self.sequence_['Facies'].to_numpy(),
                                                    z_min, z_max, dz)
        self.record_ = xr.Dataset(
            data_vars={
                'Stratigraphy': (
                    ('Grain size', 'Z', 'X'),
                    stratigraphy/dz,
                    {
                        'units': '-',
                        'description': 'fraction of each grain size in a cell',
                    },
                ),
                'Facies': (
                    ('Environment', 'Z', 'X'),
                    facies,
                    {
                        'units': '-',
                        'description': 'fraction of each facies in a cell'
                    },
                ),
            },
            coords={
                'X': np.linspace(self.spacing/2.,
                                 self.spacing*(facies.shape[2] - 0.5),
                                 facies.shape[2]),
                'Z': np.linspace(z_min + dz/2.,
                                 z_max - dz/2.,
                                 facies.shape[1]),
                'Grain size': self.sediment_size,
                'Environment': ['none', 'substratum', 'coastal plain', 'lagoon', 'barrier island', 'upper shoreface', 'lower shoreface', 'offshore'],
            },
        )
        self.record_['X'].attrs['units'] = 'meter'
        self.record_['Z'].attrs['units'] = 'meter'
        self.record_['Grain size'].attrs['units'] = 'micrometer'

        return self
        
    def summarize(self, on_record=True, to_phi=True):
        """
        Summarizes the grain-size distribution and facies.

        Parameters
        ----------
        on_record : bool, default=True
            If true, summarizes self.record_, otherwise summarizes self.sequence_.
        to_phi : bool, default=True
            If true, computes the sorting term in phi unit, otherwise computes the
            sorting term in micrometer.

        Returns
        -------
        self : object
            Simulator with extra variables in the `sequence_` or `record_` attribute.
        """
        if on_record == True:
            dataset = self.record_
        else:
            dataset = self.sequence_

        dataset['Mean grain size'] = (
            ('Z', 'X'),
            _compute_mean_grain_size(dataset['Stratigraphy'].to_numpy(), dataset['Grain size'].to_numpy()),
            {
                'units': 'micrometer',
                'description': 'mean of the grain-size distribution'
            },
        )
        dataset['Sorting term'] = (
            ('Z', 'X'),
            _compute_std_grain_size(dataset['Stratigraphy'].to_numpy(), dataset['Grain size'].to_numpy(), to_phi),
            {
                'units': 'phi' if to_phi == True else 'micrometer',
                'description': 'standard deviation of the grain-size distribution'
            },
        )
        dataset['Major facies'] = (
            ('Z', 'X'),
            np.argmax(dataset['Facies'].to_numpy(), axis=0),
            {
                'units': '',
                'description': 'major facies: 0. none, 1. substratum, 2. coastal plain, 3. lagoon, 4. barrier island, 5. upper shoreface, 6. lower shoreface, 7. offshore'
            },
        )

        return self


################################################################################
# BarSim 2.5D

@nb.jit(nopython=True, parallel=True)
def _run_multiple(initial_elevation, sea_level_curve, sediment_supply_curve, spacing,
                  run_time, dt_fair_weather, dt_storm, max_wave_height_fair_weather, allow_storms,
                  start_with_storm, max_wave_height_storm, tidal_amplitude,
                  min_tidal_area_for_transport, sediment_size, sediment_fraction,
                  initial_substratum, erodibility, washover_fraction, tide_sand_fraction,
                  depth_factor_backbarrier, depth_factor_shoreface, local_factor_shoreface,
                  local_factor_backbarrier, fallout_rate_backbarrier, fallout_rate_shoreface,
                  max_width_backbarrier, max_barrier_height_shoreface,
                  max_barrier_height_backbarrier, z_min, z_max, dz, seed):
    
    stratigraphy = np.empty((len(sediment_size),
                             int((z_max - z_min)/dz),
                             initial_elevation.shape[0],
                             initial_elevation.shape[1]))
    facies = np.empty((8,
                       int((z_max - z_min)/dz),
                       initial_elevation.shape[0],
                       initial_elevation.shape[1]))
    for i in nb.prange(initial_elevation.shape[0]):
        _, _, _, elevation, time_stratigraphy, time_facies = _run(initial_elevation[i], sea_level_curve, sediment_supply_curve[i],
                                                                  spacing, run_time, dt_fair_weather, dt_storm, max_wave_height_fair_weather,
                                                                  allow_storms, start_with_storm, max_wave_height_storm,
                                                                  tidal_amplitude, min_tidal_area_for_transport, sediment_size,
                                                                  sediment_fraction, initial_substratum, erodibility,
                                                                  washover_fraction, tide_sand_fraction, depth_factor_backbarrier,
                                                                  depth_factor_shoreface, local_factor_shoreface,
                                                                  local_factor_backbarrier, fallout_rate_backbarrier,
                                                                  fallout_rate_shoreface, max_width_backbarrier,
                                                                  max_barrier_height_shoreface, max_barrier_height_backbarrier,
                                                                  seed)
        stratigraphy[:, :, i], facies[:, :, i] = _regrid_stratigraphy(elevation[-1], time_stratigraphy,
                                                                      time_facies, z_min, z_max, dz)
        
    return stratigraphy, facies


@nb.jit(nopython=True, parallel=True)
def _summarize_multiple(stratigraphy, facies, grain_size, to_phi):
    
    mean = np.empty(stratigraphy.shape[1:])
    std = np.empty(stratigraphy.shape[1:])
    major_facies = np.empty(stratigraphy.shape[1:])
    for i in nb.prange(stratigraphy.shape[3]):
        mean[..., i] = _compute_mean_grain_size(stratigraphy[..., i], grain_size)
        std[..., i] = _compute_std_grain_size(stratigraphy[..., i], grain_size, to_phi)
        for j in nb.prange(facies.shape[2]):
            for k in nb.prange(facies.shape[1]):
                major_facies[k, j, i] = np.argmax(facies[:, k, j, i])
        
    return mean, std, major_facies


class BarSimPseudo3D:
    """
    2.5D implementation of the multiple grain-size, process-response model BarSim.

    Parameters
    ----------
    initial_elevation : array-like of shape (n_y, n_x)
        The initial elevation (m).
    sea_level_curve : array-like of shape (n_t, 2)
        The inflection points of sea level in time (yr, m).
    sediment_supply_curve : array-like of shape (n_t, 2)
        The inflection points of sediment supply in time (yr, m2/yr).
    spacing : array-like of shape (2,), default=(100., 100.)
        The grid spatial resolution (m).
    max_wave_height_fair_weather : float, default=1.5
        Maximum fair-weather wave height used to simulate a time-series of wave
        height under fair-weather conditions (m).
    allow_storms : bool, default=True
        If True, allows storm events, otherwise only fair weather is simulated.
    start_with_storm : bool, default=False
        If True, starts the simulation with a storm, otherwise starts with fair-
        weather conditions.
    max_wave_height_storm : float, default=6.
        Maximum wave height for storm events (m).
    tidal_amplitude : float, default=2.
        Tidal amplitude (m).
    min_tidal_area_for_transport : float, default=100.
        Minimum tidal prism area required for tide induced transport to the
        backbarrier/tidal basin (m).
    sediment_size : array-like of shape (n_grain_sizes,), default=(5, 50, 125, 250)
        Diameter of the grains of the different classes of sediments within the
        influx of additional sediment by longshore drift (μm).
    sediment_fraction : array-like of shape (n_grain_sizes,), default=(0.25, 0.25, 0.25, 0.25)
        Fraction of the different classes of sediments within the influx of additional
        sediment by longshore drift. Each value must be between 0 and 1.
    initial_substratum : array-like of shape (n_grain_sizes, n_x) or (2,), default=(100., (0.25, 0.25, 0.25, 0.25))
        Initial thickness of the substratum for each grain size. It can be directly
        an array representing the substratum, or a tuple (thickness, tuple with the
        proportion of each grain size), which is converted into an array during
        initialization. Make sure that the substrate is thick enough to not be
        fully eroded during simulation.
    erodibility : float, default=0.1
        Multiplier or tuning parameter (c_e in equation 3 of Storms (2003)) to
        adjust the erosion capacity.
    washover_fraction : float, default=0.5
        Fraction of sediments going into washover instead of shoreface. It must
        be between 0 and 1.
    tide_sand_fraction : float, default=0.3
        Fraction of the sand-prone grain-size class available for tide-induced
        transport. This rule ensures that tide-induced transport is focused on
        the fine-grained fractions by limiting the sand-sized grain-size
        fraction. NOTE that the correct grain-size class is limited! It must be
        between 0 and 1.
    depth_factor_backbarrier : float, default=5.
        Parameter `A` in equation 9 (Storms et al., 2002) applied to backbarrier
        deposition.
    depth_factor_shoreface : float, default=10.
        Parameter `A` in equation 9 (Storms et al., 2002) applied to backbarrier
        deposition.
    local_factor_shoreface : float, default=1.5
        Multiplier or tuning parameter (c_g in equations 13 and 14 of Storms (2003))
        to adjust for local conditions of the sediment travel distance at the
        shoreface.
    local_factor_backbarrier : float, default=1.
        Multiplier or tuning parameter (c_g in equations 13 and 14 of Storms (2003))
        to adjust for local conditions of the sediment travel distance at the
        backbarrier.
    fallout_rate_backbarrier : float, default=0.
        Constant fall-out rate of fines (the finest sediment fraction only) in the
        backbarrier area (mm/yr). Is only active during fair-weather conditions.
    fallout_rate_shoreface : float, default=0.0002
        Constant fall-out rate of fines (the finest sediment fraction only) in the
        shoreface-shelf (mm/yr), representing pelagic sedimentation. It is only
        active during fair-weather conditions.
    max_width_backbarrier : float, default=500.
        Maximum width of the backbarrier (m).
    max_barrier_height_shoreface : float, default=1.
        Maximum height of barrier island on the seaward side (m). Higher elevations
        above sea level limit the deposition. The parameter refers to z_l in
        equation 9 in Storms et al. (2002). This is a sensitive parameter, better
        keep the default value.
    max_barrier_height_backbarrier : float, default=1.
        Maximum height of barrier island on the backbarrier side (m). Higher
        elevations above sea level limit the deposition. The parameter refers to
        z_l in equation 9 in Storms et al. (2002). This is a sensitive parameter,
        better keep the default value.
    preinterpolate_curves: bool, default=False
        If True, pre-interpolates the sea level and sediment supply curves.
    monotonic: bool, default=True
        If True, uses a monotonic interpolation for sea level and sediment supply,
        otherwise uses a cubic interpolation.
    seed : int, default=42
        Controls random number generation for reproducibility.

    Attributes
    ----------
    sequence_ : xarray.Dataset
        Dataset containing the evolution of the elevation and stratigraphy through time.
        
    References
    ----------
    Storms, J.E.A., Weltje, G.J., Van Dijke, J.J., Geel, C.R., Kroonenberg, S.B. (2002) https://doi.org/10.1306/052501720226
        Process-response modeling of wave-dominated coastal systems: simulating evolution and stratigraphy on geological timescales.
    Storms, J.E.A. (2003) https://doi.org/10.1016/S0025-3227(03)00144-0
        Event-based stratigraphic simulation of wave-dominated shallow-marine environments
    """
    def __init__(self,
                 initial_elevation,
                 sea_level_curve,
                 sediment_supply_curve,
                 spacing=(100., 100.),
                 max_wave_height_fair_weather=1.5,
                 allow_storms=True,
                 start_with_storm=False,
                 max_wave_height_storm=6.,
                 tidal_amplitude=2.,
                 min_tidal_area_for_transport=100.,
                 sediment_size=(5., 50., 125., 250.),
                 sediment_fraction=(0.25, 0.25, 0.25, 0.25),
                 initial_substratum=(200., (0.25, 0.25, 0.25, 0.25)),
                 erodibility=0.1,
                 washover_fraction=0.5,
                 tide_sand_fraction=0.3,
                 depth_factor_backbarrier=5.,
                 depth_factor_shoreface=10.,
                 local_factor_shoreface=1.5,
                 local_factor_backbarrier=1.,
                 fallout_rate_backbarrier=0.,
                 fallout_rate_shoreface=0.0002,
                 max_width_backbarrier=500.,
                 max_barrier_height_shoreface=1.,
                 max_barrier_height_backbarrier=1.,
                 preinterpolate_curves=False,
                 monotonic=True,
                 seed=42):
        
        self.initial_elevation = np.asarray(initial_elevation)
        self.sea_level_curve = np.asarray(sea_level_curve)
        self.sediment_supply_curve = np.asarray(sediment_supply_curve)
        self.spacing = spacing
        self.max_wave_height_fair_weather = max_wave_height_fair_weather
        self.allow_storms = allow_storms
        self.start_with_storm = start_with_storm
        self.max_wave_height_storm = max_wave_height_storm
        self.tidal_amplitude = tidal_amplitude
        self.min_tidal_area_for_transport = min_tidal_area_for_transport
        self.sediment_size = np.asarray(sediment_size)
        self.sediment_fraction = np.asarray(sediment_fraction)
        if (len(initial_substratum) == 2
            and len(initial_substratum[1]) == len(sediment_size)):
            self.initial_substratum = np.full((len(sediment_size), initial_elevation.shape[1]),
                                              initial_substratum[0])
            self.initial_substratum *= np.asarray(initial_substratum[1])[:, np.newaxis]
        else:
            self.initial_substratum = np.asarray(initial_substratum)
        self.erodibility = erodibility
        self.washover_fraction = washover_fraction
        self.tide_sand_fraction = tide_sand_fraction
        self.depth_factor_backbarrier = depth_factor_backbarrier
        self.depth_factor_shoreface = depth_factor_shoreface
        self.local_factor_shoreface = local_factor_shoreface
        self.local_factor_backbarrier = local_factor_backbarrier
        self.fallout_rate_backbarrier = fallout_rate_backbarrier
        self.fallout_rate_shoreface = fallout_rate_shoreface
        self.max_width_backbarrier = max_width_backbarrier
        self.max_barrier_height_shoreface = max_barrier_height_shoreface
        self.max_barrier_height_backbarrier = max_barrier_height_backbarrier
        self.preinterpolate_curves = preinterpolate_curves
        self.monotonic = monotonic
        self.seed = seed
        
    def _preinterpolate_curves(self, run_time, dt_fair_weather, dt_storm):
        """
        Preinterpolate the sea level and sediment supply curves using a cubic interpolation
        before using them in Numba.
        TODO: Would be better to have a cubic interpolation in Numba.
        """
        if self.allow_storms == True:
            n_time_steps = math.ceil(2*run_time/(dt_fair_weather + dt_storm)) + 1
        else:
            n_time_steps = math.ceil(run_time/dt_fair_weather) + 1
        
        if self.monotonic == False:
            sea_level_function = interpolate.interp1d(self.sea_level_curve[:, 0],
                                                      self.sea_level_curve[:, 1],
                                                      kind='cubic')
        else:
            sea_level_function = interpolate.PchipInterpolator(self.sea_level_curve[:, 0],
                                                               self.sea_level_curve[:, 1])
        sea_level_interp = np.empty((n_time_steps, 2))
        sea_level_interp[:, 0] = np.linspace(0., run_time, n_time_steps)
        sea_level_interp[:, 1] = sea_level_function(sea_level_interp[:, 0])
        
        sediment_supply_interp = np.empty((len(self.sediment_supply_curve), n_time_steps, 2))
        for i in range(len(self.sediment_supply_curve)):
            if self.monotonic == False:
                sediment_supply_function = interpolate.interp1d(self.sediment_supply_curve[i, :, 0],
                                                                self.sediment_supply_curve[i, :, 1],
                                                                kind='cubic')
            else:
                sediment_supply_function = interpolate.PchipInterpolator(self.sediment_supply_curve[i, :, 0],
                                                                         self.sediment_supply_curve[i, :, 1])
            sediment_supply_interp[i, :, 0] = sea_level_interp[:, 0]
            sediment_supply_interp[i, :, 1] = sediment_supply_function(sediment_supply_interp[i, :, 0])
        
        return sea_level_interp, sediment_supply_interp
    
    def run(self, z_min, z_max, dz, run_time=10000., dt_fair_weather=15., dt_storm=1.):
        """
        Runs BarSim for a given run time.

        Parameters
        ----------
        z_min : float
            Lower border of the grid (m).
        z_max : float
            Upper border of the grid (m).
        dz : float
            Resolution of the grid (m).
        run_time : float, default=10000.
            Duration of the run (yr).
        dt_fair_weather : float, default=15.
            Time step in fair weather conditions (yr).
        dt_storm : float, default=1.
            Time step during a storm (yr).

        Returns
        -------
        self : object
            Simulator with a `record_` attribute.
        """
        if self.preinterpolate_curves == True:
            sea_level_curve, sediment_supply_curve = self._preinterpolate_curves(run_time, dt_fair_weather, dt_storm)
        else:
            sea_level_curve, sediment_supply_curve = self.sea_level_curve, self.sediment_supply_curve
            
        stratigraphy, facies = _run_multiple(self.initial_elevation, sea_level_curve, sediment_supply_curve,
                                             self.spacing[1], run_time, dt_fair_weather, dt_storm, self.max_wave_height_fair_weather,
                                             self.allow_storms, self.start_with_storm, self.max_wave_height_storm,
                                             self.tidal_amplitude, self.min_tidal_area_for_transport, self.sediment_size,
                                             self.sediment_fraction, self.initial_substratum, self.erodibility,
                                             self.washover_fraction, self.tide_sand_fraction, self.depth_factor_backbarrier,
                                             self.depth_factor_shoreface, self.local_factor_shoreface,
                                             self.local_factor_backbarrier, self.fallout_rate_backbarrier,
                                             self.fallout_rate_shoreface, self.max_width_backbarrier,
                                             self.max_barrier_height_shoreface, self.max_barrier_height_backbarrier,
                                             z_min, z_max, dz, self.seed)
        self.record_ = xr.Dataset(
            data_vars={
                'Stratigraphy': (
                    ('Grain size', 'Z', 'Y', 'X'),
                    stratigraphy/dz,
                    {'units': '-', 'description': 'fraction of each grain size in a cell'},
                ),
                'Facies': (
                    ('Environment', 'Z', 'Y', 'X'),
                    facies,
                    {'units': '-', 'description': 'fraction of each facies in a cell'},
                    ),
            },
            coords={
                'X': np.linspace(self.spacing[1]/2.,
                                 self.spacing[1]*(stratigraphy.shape[3] - 0.5),
                                 stratigraphy.shape[3]),
                'Y': np.linspace(self.spacing[0]/2.,
                                 self.spacing[0]*(stratigraphy.shape[2] - 0.5),
                                 stratigraphy.shape[2]),
                'Z': np.linspace(z_min + dz/2.,
                                 z_max - dz/2.,
                                 stratigraphy.shape[1]),
                'Grain size': self.sediment_size,
                'Environment': ['none', 'substratum', 'coastal plain', 'lagoon', 'barrier island', 'upper shoreface', 'lower shoreface', 'offshore'],
            },
        )
        self.record_['X'].attrs['units'] = 'meter'
        self.record_['Y'].attrs['units'] = 'meter'
        self.record_['Z'].attrs['units'] = 'meter'
        self.record_['Grain size'].attrs['units'] = 'micrometer'

        return self
        
    def summarize(self, to_phi=True):
        """
        Summarizes the grain-size distribution and facies.

        Parameters
        ----------
        to_phi : bool, default=True
            If true, computes the sorting term in phi unit, otherwise computes the
            sorting term in micrometer.

        Returns
        -------
        self : object
            Simulator with extra variables in the `record_` attribute.
        """
        mean, std, major_facies = _summarize_multiple(self.record_['Stratigraphy'].to_numpy(),
                                                      self.record_['Facies'].to_numpy(),
                                                      self.record_['Grain size'].to_numpy(),
                                                      to_phi)

        self.record_['Mean grain size'] = (
            ('Z', 'Y', 'X'),
            mean,
            {
                'units': 'micrometer',
                'description': 'mean of the grain-size distribution',
            }
        )
        self.record_['Sorting term'] = (
            ('Z', 'Y', 'X'),
            std,
            {
                'units': 'phi' if to_phi == True else 'micrometer',
                'description': 'standard deviation of the grain-size distribution',
            }
        )
        self.record_['Major facies'] = (
            ('Z', 'Y', 'X'),
            major_facies,
            {
                'units': '',
                'description': 'major facies: 0. none, 1. substratum, 2. coastal plain, 3. lagoon, 4. barrier island, 5. upper shoreface, 6. lower shoreface, 7. offshore',
            }
        )

        return self
        
    def mesh(self, zscale=1.):
        """
        Turns `record_` into a PyVista mesh object.
        
        Parameters
        ----------
        zscale : float, default=1.
            Vertical scale factor.

        Returns
        -------
        mesh : pyvista.StructuredGrid
            The mesh.
        """
        x = self.record_['X'].to_numpy().copy()
        x -= (x[1] - x[0])/2.
        x = np.append(x, x[-1] + (x[1] - x[0]))
        y = self.record_['Y'].to_numpy().copy()
        y -= (y[1] - y[0])/2.
        y = np.append(y, y[-1] + (y[1] - y[0]))
        z = self.record_['Z'].to_numpy().copy()
        z -= (z[1] - z[0])/2.
        z = np.append(z, z[-1] + (z[1] - z[0]))
        xx, yy, zz = np.meshgrid(x, y, zscale*z, indexing='ij')

        mesh = pv.StructuredGrid(xx, yy, zz)
        median = self.record_['Mean grain size'].to_numpy().copy()
        median[(self.record_['Facies'][0] == 1) | (self.record_['Facies'][1] == 1)] = np.nan
        mesh['Mean grain size'] = median.ravel()
        major_facies = self.record_['Major facies'].to_numpy().astype(float)
        major_facies[(self.record_['Facies'][0] > 0) | (self.record_['Facies'][1] > 0)] = np.nan
        mesh['Major facies'] = major_facies.ravel()
        
        return mesh
