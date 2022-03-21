"""BarSim"""

# LICENCE GOES HERE


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
def _initialize_fluxes(TDcorr_SF, TDcorr_OW, texture):
    """
    Adopted from Guillen and Hoekstra 1996, 1997, but slightly different (this
    is only for the >0.125mm fraction)
    """
    flux_shoreface_basic = np.zeros(len(texture))
    flux_overwash_basic = np.zeros(len(texture))
    for j in range (len(texture)):
        flux_shoreface_basic[j] = TDcorr_SF * (110 + 590 * ((0.125 / (texture[j] * 0.001)) ** 2.5))
        flux_overwash_basic[j] = TDcorr_OW * (110 + 590 * ((0.125 / (texture[j] * 0.001)) ** 2.5))
        
    return flux_shoreface_basic, flux_overwash_basic


@nb.jit(nopython=True)
def _compute_Theta_wl(texture):
    """
    TODO: Rewrite this function, not great
    """
    Dd = np.zeros(len(texture))
    for j in range(len(texture)):
        Dd[j] = (texture[j]/1000) * (1 * 981 * 2 / 0.01**2)**(1/3)    
    Wd = np.zeros(len(texture))
    for j in range(len(texture)):
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
    Theta_wl = np.zeros(len(texture))
    for j in range(len(texture)):
        if Wd[j] > 0:
            Theta_wl[j] = 0.0246 * Wd[j]**(-0.55)
                         
    return Theta_wl


@nb.jit(nopython=True)
def _correct_travel_distance(mode, event, flux_shoreface_basic, flux_overwash_basic,
                             W, n_grain_sizes, correction_factor=0.25):
    """
    Corrects travel distance for storm sediment transport processes (e.g. downwelling).
    """
    flux_shoreface = np.zeros(n_grain_sizes)
    flux_overwash = np.zeros(n_grain_sizes)
    if mode == 1: #no events
        for i in range (n_grain_sizes):
            flux_shoreface[i] = flux_shoreface_basic[i]
            flux_overwash[i] = flux_overwash_basic[i]
    elif mode == 2: # events
        if (event is True):
            correction_shoreface = correction_factor*W
            correction_backbarrier = correction_factor*W
            for i in range (n_grain_sizes):
                flux_shoreface[i] = flux_shoreface_basic[i]*correction_shoreface
                flux_overwash[i] = flux_overwash_basic[i]*correction_backbarrier
        else:
            for i in range (n_grain_sizes):
                flux_shoreface[i] = flux_shoreface_basic[i]
                flux_overwash[i] = flux_overwash_basic[i]
                
    return flux_shoreface, flux_overwash


@nb.jit(nopython=True)
def _event(mode, event, time, W_event, W_fw, dt_min, dt_fw, flux_shoreface_basic,
           flux_overwash_basic, n_grain_sizes):
    """
    Event appraoch
    TODO: What is this function doing?
    """
    if mode == 1: #no events
        W = W_event
        T = 5
        dt = dt_min
        time += dt
    elif mode == 2: # events
        if (event is True):
            W = W_event + 2.*random.random()
            T = 4 + 0.5*random.random()
            # wavebase_depth = T * 0.5
            dt = dt_min
            time += dt
        else:
            W = W_fw + 2.*random.random()
            T = 2.5 + 0.5*random.random()
            # wavebase_depth = T * 0.5
            dt = dt_fw
            time += dt

    flux_shoreface, flux_overwash = _correct_travel_distance(mode,
                                                             event,
                                                             flux_shoreface_basic,
                                                             flux_overwash_basic,
                                                             W,
                                                             n_grain_sizes)
    
    return W, T, dt, time, flux_shoreface, flux_overwash


@nb.jit(nopython=True)
def _compute_orbital_velocity(elevation, sea_level, W, T, Theta_wl, texture, i_coastline):
    """
    Computes the actual horizontal orbital velocity (m/s) based on Komar, Beach
    processes, 2nd edition, p163 and 164 and the max. horizontal orbital velocity
    (m/s) for each grain size class based on Le Roux (2001, eq. 39, unit = cm/s).
    """
    L_int = np.zeros(len(elevation))
    L_deep = (9.81 * T**2) / (2*3.1428)  #calculate deep water wave length
    for i in range(i_coastline + 1, len(elevation)):
        tmp_double = 2*3.1428*(sea_level - elevation[i])/L_deep
        L_int[i] = L_deep*math.sqrt(math.tanh(tmp_double))
    orbital_velocity = np.zeros(len(elevation))
    for i in range(i_coastline + 1, len(elevation)):
        if L_int[i] != 0.:
            num = T * math.sinh(2 * 3.1428 * (sea_level - elevation[i]) / L_int[i])
            if num != 0.:
                orbital_velocity[i] = 3.1428*W/num
    orbital_velocity[i_coastline] = 20 # what is this number?

    orbital_velocity_max = np.zeros(len(texture) + 1)
    for i in range(len(texture)):
        orbital_velocity_max[i] = (-0.01 * ((Theta_wl[i] * 981 * texture[i] * 0.0001 * 2) ** 2 / (1 * 0.01 / T)) + 1.3416 * ((Theta_wl[i] * 981 * texture[i] * 0.0001 * 2) / ((1 * 0.01) / T) ** 0.5) - 0.6485) / 100.
        
    return orbital_velocity, orbital_velocity_max


@nb.jit(nopython=True)
def _decompose_domain(elevation, sea_level, W, T, texture, Theta_wl):
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
                                                                       W, T, Theta_wl, texture,
                                                                       i_coastline)
    while orbital_velocity[i_wavebase] <= orbital_velocity_max[0] and i_wavebase > 0:  #for the finest fraction
        i_wavebase -= 1
            
    return i_mainland, i_backbarrier, i_coastline, i_wavebase


@nb.jit(nopython=True)
def _classify_facies(event, i_mainland, i_backbarrier, i_coastline, i_wavebase,
                     i_wavebase_event, i_wavebase_fw, n_x):
    """
    Classifies the domain into facies.
    """
    if event == False:
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
def _erode(elevation, stratigraphy, sea_level, erodibility, W, W_fw, i_time,
           i_coastline, i_wavebase):
    """
    Erodes the domain based on Storms (2003) without reflection correction cd(t).
    """
    erosion_total = np.zeros(len(elevation[i_time]))
    for i in range(i_coastline, i_wavebase + 1):
        erosion_total[i] = erodibility * (W/W_fw) * ((elevation[i_time, i] - elevation[i_time, i_wavebase]) / (sea_level - elevation[i_time, i_wavebase]))**3
        
    stratigraphy, erosion = _erode_stratigraphy(stratigraphy, erosion_total, 
                                                i_time, i_coastline, i_wavebase)
        
    return stratigraphy, erosion


@nb.jit(nopython=True)
def _distribute_fluxes(erosion, sediment_supply, sea_level, sea_level_prev, ow_part,
                       BB_max_width, texture_ratio, i_mainland, i_backbarrier,
                       i_coastline, i_wavebase, dt, spacing):
    """
    Distributes the fluxes along the domain.
    """
    n_grain_sizes = len(texture_ratio)
    total_flux = np.zeros(n_grain_sizes)
    for i in range(i_coastline, i_wavebase + 1):
        for j in range(n_grain_sizes): 
            total_flux[j] += erosion[j, i]*spacing
    for j in range(n_grain_sizes):
        total_flux[j] += sediment_supply*texture_ratio[j]*dt

    total_flux_washover = np.zeros(n_grain_sizes)
    total_flux_shoreface = np.zeros(n_grain_sizes)
    if (i_backbarrier > 1
        and i_coastline - i_mainland > 0
        and (i_coastline - i_backbarrier)*spacing < BB_max_width
        and sea_level >= sea_level_prev
        and ow_part > 0.): 
        for j in range(2, n_grain_sizes):    #NOTE!! ASSUME GRAINSIZE 3 and 4 ARE SAND
            total_flux_washover[j] = ow_part*total_flux[j]
        for j in range(n_grain_sizes):
            total_flux_shoreface[j] = total_flux[j] - total_flux_washover[j]
            
    return total_flux, total_flux_washover, total_flux_shoreface


@nb.jit(nopython=True)
def _deposit_washover(elevation, total_flux_washover, flux_overwash, sea_level,
                      sea_level_prev, ow_part, BB_max_width, A_factor_BB, max_height_BB,
                      i_time, i_mainland, i_backbarrier, i_coastline, spacing):
    """
    Deposits washover sediments based on the restriction algorithm (travel distance ~ expected height).
    """
    n_grain_sizes = len(total_flux_washover)
    sediment_flux = np.zeros((n_grain_sizes, len(elevation[i_time])))
    deposition_washover = np.zeros((n_grain_sizes, len(elevation[i_time])))
    if (i_backbarrier > 1
        and i_coastline - i_mainland > 0
        and (i_coastline - i_backbarrier)*spacing < BB_max_width
        and sea_level >= sea_level_prev
        and ow_part > 0.):
        for j in range(2, n_grain_sizes):
            sediment_flux[j, i_coastline - 1] = total_flux_washover[j]

        for i in range(i_coastline - 1, i_mainland, -1):
            H_norm = (elevation[i_time, i] - sea_level)/max_height_BB

            f_add = np.zeros(n_grain_sizes)
            for j in range(2, n_grain_sizes):
                f_add[j] = flux_overwash[j]*(1. + 2.71828283**(H_norm*A_factor_BB))
                sediment_flux[j, i - 1] = sediment_flux[j, i] - sediment_flux[j, i]*spacing/f_add[j]  
                deposition_washover[j, i] = (sediment_flux[j, i] - sediment_flux[j, i - 1])/spacing
                
    return sediment_flux, deposition_washover


@nb.jit(nopython=True)
def _redistribute_fluxes(sediment_flux, total_flux, total_flux_shoreface, sea_level,
                         sea_level_prev, ow_part, BB_max_width, i_backbarrier,
                         i_coastline, i_mainland, spacing):
    """
    Redistributes fluxes in the domain.
    """
    n_grain_sizes = sediment_flux.shape[0]
    if (i_backbarrier > 1
        and i_coastline - i_mainland > 0
        and (i_coastline - i_backbarrier)*spacing < BB_max_width
        and sea_level >= sea_level_prev
        and ow_part > 0.):
        for j in range(n_grain_sizes):
            sediment_flux[j, i_coastline] = sediment_flux[j, i_mainland] + total_flux_shoreface[j]
    else:
        for j in range(n_grain_sizes):
            sediment_flux[j, i_coastline] = total_flux[j]
        
    return sediment_flux


@nb.jit(nopython=True)
def _deposit_tidal(elevation, deposition_washover, sediment_flux, sea_level, Tidalampl,
                   Tide_Acc_VAR, TidalSand, fallout_rate_bb, dh_ratio_acc, event,
                   i_time, i_backbarrier, i_mainland, i_coastline, dt, spacing):
    """
    Deposits tidal sediments.
    """
    n_grain_sizes = sediment_flux.shape[0]
    deposition_tidal = np.zeros((n_grain_sizes, len(elevation[i_time])))
    deposition_tidal_tot = np.zeros(n_grain_sizes)
    
    sediment_flux_total = 0.
    for j in range(n_grain_sizes):
        sediment_flux_total += sediment_flux[j, i_coastline]
    
    if event == False and Tidalampl > 0:
        
        deposition_tidal_sum = 0.
        BB_acc = 0
        # TODO: should that be changed for not taking into account if sea_level - elevation[i_time, i] <= 0?
        len_backbarrier = 0
        # Determine backbarrier accommodation
        for i in range(i_backbarrier - 1, i_mainland, -1):
            if sea_level - elevation[i_time, i] > 0.:
                BB_acc += (sea_level - elevation[i_time, i])*spacing
                len_backbarrier += 1
        # Minimum tidal basin size: USE ASMITA HERE? Area that is wet is 
        # proportional to the Tidal Amplitude. Also assume minimal hydraulic
        # gradient befor tidal processes become active
        if BB_acc < Tide_Acc_VAR - Tidalampl*spacing*len_backbarrier:
            tidal_supply = False
        else:
            tidal_supply = True
            
        # Max deposition rate based on Tidal amplitude, Backbarrier width and
        # time step, limited by total sediment flux
        if tidal_supply == True and Tidalampl > 0:
            dh_Tidal_cap = len_backbarrier*spacing*Tidalampl*dt*0.001 # 1 is tuning parameter

            if (dh_Tidal_cap > sediment_flux_total):
                dh_Tidal_cap = sediment_flux_total

            if (dh_Tidal_cap > BB_acc):
                dh_ratio_acc = 1
            else:
                dh_ratio_acc = dh_Tidal_cap/BB_acc

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
                    deposition_tidal[j, i] = dh_ratio_acc*(sea_level - elevation[i_time, i])*deposition_tidal_tot[j]/deposition_tidal_sum #grain size distribution tidal deposits (sum should be 1)
                # Coarsest fraction less
                deposition_tidal[n_grain_sizes - 1, i] = TidalSand*dh_ratio_acc*(sea_level - elevation[i_time, i])*deposition_tidal_tot[n_grain_sizes - 1]/deposition_tidal_sum   #grain size distribution tidal deposits (sum should be 1)
            
            # Leftover sediment for SHOREFACE deposition
            for j in range(n_grain_sizes):
                sediment_flux[j, i_coastline] -= deposition_tidal[j, i]*spacing

    # Update layer thickness (dh) and cellheight and assign remaining sediment for shoreface sedimentation
    if (fallout_rate_bb > 0):
        for i in range(i_backbarrier - 1, i_mainland, -1): #fallout rate in backbarrier (organics + fines)
            if sea_level - elevation[i_time, i] > 0.:
                if fallout_rate_bb*dt < (sea_level - elevation[i_time, i]):                 #in case of accommodation
                    deposition_washover[0, i] += fallout_rate_bb*dt
                else:
                    deposition_washover[0, i] += (sea_level - elevation[i_time, i]) #no BB deposition above SL
            
    return sediment_flux, deposition_washover, deposition_tidal


@nb.jit(nopython=True)
def _deposit_shoreface(elevation, sediment_flux, flux_shoreface, sea_level, max_height_SF,
                       fallout_rate_sf, A_factor_SF, i_time, i_coastline, spacing, dt):
    """
    Deposits shoreface sediments.
    """
    n_grain_sizes = sediment_flux.shape[0]
    deposition_shoreface = np.zeros((n_grain_sizes, len(elevation[i_time])))
    # TODO: Would it be possible to do sediment_flux[j, i - 1] - sediment_flux[j, i]?
    for i in range(i_coastline, len(elevation[i_time]) - 1):
        H_norm = (elevation[i_time, i] - sea_level)/max_height_SF - max_height_SF
        f_add = np.zeros(n_grain_sizes)
        for j in range(n_grain_sizes):
            f_add[j] = flux_shoreface[j]*(1 + 2.71828283**(H_norm*A_factor_SF))
        for j in range(n_grain_sizes):
            sediment_flux[j, i + 1] = sediment_flux[j, i] - sediment_flux[j, i]*spacing/f_add[j]
        for j in range(n_grain_sizes):
            deposition_shoreface[j, i] = (sediment_flux[j, i] - sediment_flux[j, i + 1])/spacing
        deposition_shoreface[0, i] += fallout_rate_sf*dt
    
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
             flux_shoreface, sea_level, ow_part, BB_max_width, A_factor_BB,
             max_height_BB, Tidalampl, Tide_Acc_VAR, TidalSand, fallout_rate_bb,
             max_height_SF, fallout_rate_sf, A_factor_SF, dh_ratio_acc, event,
             texture_ratio, i_time, i_mainland, i_backbarrier, i_coastline,
             i_wavebase, dt, spacing):
    """
    Deposits sediments in the domain.
    """
    total_flux, total_flux_washover, total_flux_shoreface = _distribute_fluxes(erosion, sediment_supply[i_time],
                                                                               sea_level[i_time], sea_level[i_time - 1],
                                                                               ow_part, BB_max_width, texture_ratio,
                                                                               i_mainland, i_backbarrier,
                                                                               i_coastline, i_wavebase,
                                                                               dt, spacing)
    sediment_flux, deposition_washover = _deposit_washover(elevation, total_flux_washover, flux_overwash,
                                                           sea_level[i_time], sea_level[i_time - 1], ow_part,
                                                           BB_max_width, A_factor_BB, max_height_BB, i_time,
                                                           i_mainland, i_backbarrier, i_coastline, spacing)
    sediment_flux = _redistribute_fluxes(sediment_flux, total_flux, total_flux_shoreface,
                                         sea_level[i_time], sea_level[i_time - 1], ow_part, BB_max_width,
                                         i_backbarrier, i_coastline, i_mainland, spacing)
    sediment_flux, deposition_washover, deposition_tidal = _deposit_tidal(elevation, deposition_washover,
                                                                          sediment_flux, sea_level[i_time],
                                                                          Tidalampl, Tide_Acc_VAR, TidalSand,
                                                                          fallout_rate_bb, dh_ratio_acc, event,
                                                                          i_time, i_backbarrier, i_mainland,
                                                                          i_coastline, dt, spacing)
    sediment_flux, deposition_shoreface = _deposit_shoreface(elevation, sediment_flux, flux_shoreface,
                                                             sea_level[i_time], max_height_SF, fallout_rate_sf,
                                                             A_factor_SF, i_time, i_coastline, spacing, dt)
    
    elevation = _update_elevation(elevation, erosion, deposition_washover, deposition_tidal,
                                  deposition_shoreface, i_time, i_mainland)
    stratigraphy = _deposit_stratigraphy(stratigraphy, deposition_washover, deposition_tidal,
                                         deposition_shoreface, i_time, i_mainland)
    
    return elevation, stratigraphy
    

@nb.jit(nopython=True)
def _run(initial_elevation,
         initial_substratum,
         sea_level_curve,
         sediment_supply_curve,
         duration=10000.,
         spacing=100.,
         ow_part=0.5,
         mode=2,
         W_fw=1.5,
         W_event=6.,
         dt_min=1,
         dt_fw=15,
         Tidalampl=2,
         Tide_Acc_VAR=100,
         TidalSand=0.3,
         dh_ratio_acc=200000,
         TDcorr_SF=1.5,
         TDcorr_OW=1.,
         erodibility=0.1,
         BB_max_width=500,
         A_factor_BB=5,
         A_factor_SF=10,
         max_height_SF=1.,
         max_height_BB=1.,
         substrate_erosion=1,
         fallout_rate_bb=0.,
         fallout_rate_sf=0.0002,
         texture=(5., 50., 125., 250.),
         texture_ratio=(0.25, 0.25, 0.25, 0.25),
         event=False,
         seed=42):
    """
    Runs BarSim for a given duration.
    """
    random.seed(seed)
    
    dt_average = (dt_min + dt_fw)//2 - 1 if mode == 2 else dt_min
    n_time_steps = int(duration/dt_average) + 1
    
    time = np.zeros(n_time_steps)
    sea_level = np.zeros(n_time_steps)
    sea_level[0] = sea_level_curve[0, 1]
    sediment_supply = np.zeros(n_time_steps)
    sediment_supply[0] = sediment_supply_curve[0, 1]
    elevation = np.zeros((n_time_steps, len(initial_elevation)))
    elevation[0] = initial_elevation    
    stratigraphy = np.zeros((len(texture), n_time_steps, len(initial_elevation)))
    stratigraphy[:, 0] = initial_substratum
    facies = np.ones((n_time_steps, len(initial_elevation)), np.int8)

    flux_shoreface_basic, flux_overwash_basic = _initialize_fluxes(TDcorr_SF, TDcorr_OW, texture)
    Theta_wl = _compute_Theta_wl(texture)
    i_wavebase_event = 0
    i_wavebase_fw = 0

    # TODO: What is that parameter? Why is it an input as well?
    event = False

    i_time = 1
    while time[i_time - 1] < duration:
        
        W, T, dt, time[i_time], flux_shoreface, flux_overwash = _event(mode, event, time[i_time - 1], W_event, W_fw, dt_min, dt_fw, flux_shoreface_basic, flux_overwash_basic, len(texture))

        sea_level[i_time] = np.interp(time[i_time],
                                      sea_level_curve[:, 0],
                                      sea_level_curve[:, 1])
        sediment_supply[i_time] = np.interp(time[i_time],
                                            sediment_supply_curve[:, 0],
                                            sediment_supply_curve[:, 1])
        elevation[i_time] = elevation[i_time - 1]
        
        i_mainland, i_backbarrier, i_coastline, i_wavebase = _decompose_domain(elevation[i_time],
                                                                               sea_level[i_time], W, T,
                                                                               texture, Theta_wl)
        facies[i_time], i_wavebase_event, i_wavebase_fw = _classify_facies(event, i_mainland, i_backbarrier,
                                                                           i_coastline, i_wavebase, i_wavebase_event,
                                                                           i_wavebase_fw, len(elevation[i_time]))
        stratigraphy, erosion = _erode(elevation, stratigraphy, sea_level[i_time], erodibility,
                                       W, W_fw, i_time, i_coastline, i_wavebase)        
        elevation, stratigraphy = _deposit(elevation, stratigraphy, erosion, sediment_supply,
                                           flux_overwash, flux_shoreface, sea_level,
                                           ow_part, BB_max_width, A_factor_BB, max_height_BB,
                                           Tidalampl, Tide_Acc_VAR, TidalSand, fallout_rate_bb,
                                           max_height_SF, fallout_rate_sf, A_factor_SF, dh_ratio_acc,
                                           event, texture_ratio, i_time, i_mainland, i_backbarrier,
                                           i_coastline, i_wavebase, dt, spacing)

        i_time += 1
        if(i_time%2 == 0):
            event = True
        else:
            event = False
    
    return (time[:i_time], sea_level[:i_time], sediment_supply[:i_time],
            elevation[:i_time], stratigraphy[:, :i_time], facies[:i_time])


@nb.jit(nopython=True)
def _compute_median_grain_size(stratigraphy, texture):
    """
    """
    median = np.zeros(stratigraphy.shape[1:])
    for k in range(stratigraphy.shape[1]):
        for i in range(stratigraphy.shape[2]):
            stratigraphy_total = np.sum(stratigraphy[:, k, i])
            if stratigraphy_total > 0.:
                for j in range(stratigraphy.shape[0]):
                    median[k, i] += texture[j]*stratigraphy[j, k, i]/stratigraphy_total
                
    return median


@nb.jit(nopython=True)
def _regrid_stratigraphy(elevation, time_stratigraphy, time_facies, z_min, z_max, z_step):
    """
    Interpolates the time stratigraphy on a regular grid in space.
    """
    z_corner = np.linspace(z_min, z_max, int((z_max - z_min)/z_step) + 1)
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
                ratio = (min(z_corner[k + 1], layer_top) - max(z_corner[k], layer_bottom))/z_step
                facies[time_facies[l, i], k, i] += ratio
            if z_corner[k + 1] > elevation[i]:
                facies[0, k, i] = (z_corner[k + 1] - max(elevation[i], z_corner[k]))/z_step
            if l == 0 and z_corner[k] < layer_bottom:
                facies[0, k, i] = (min(z_corner[k + 1], layer_bottom) - z_corner[k])/z_step
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
    
    TODO: Rename input parameters, pick good defaults, reorder, and fill the doc.

    Parameters
    ----------
    initial_elevation : array-like of shape (n_x,)
        The initial elevation.
    sea_level_curve : array-like of shape (n_t, 2)
        The inflexion points of sea level in time.
    sediment_supply_curve : array-like of shape (n_t, 2)
        The inflexion points of sediment supply in time.
    spacing : float, default=100.
        The grid resolution.
    ow_part : float, default=0.5
        ???
    mode : int, default=2
        ???
    W_fw : float, default=1.5
        ???
    W_event : float, default=6.
        ???
    Tidalampl : ???, default=2
        ???
    Tide_Acc_VAR : ???, default=100
        ???
    TidalSand : float, default=0.3
        ???
    dh_ratio_acc : ???, default=200000
        ??? Why a default value so high? When the default value is actually used it leads to crazy results
    TDcorr_SF : float, default=1.5
        ???
    TDcorr_OW : float, default=1.
        ???
    erodibility : float, default=0.1
        ???
    BB_max_width : ???, default=500
        ???
    A_factor_BB : ???, default=5
        ???
    A_factor_SF : ???, default=10
        ???
    A_factor_SF : ???, default=10
        ???
    max_height_SF : float, default=1.
        ???
    max_height_BB : float, default=1.
        ???
    substrate_erosion : ???, default=1
        ??? That parameter is never used
    fallout_rate_bb : float, default=0.
        ???
    fallout_rate_sf : float, default=0.0002
        ???
    texture : array-like of shape (n_grain_sizes,), default=(5, 50, 125, 250)
        ???
    texture_ratio : array-like of shape (n_grain_sizes,), default=(0.25, 0.25, 0.25, 0.25)
        ???
    initial_substratum : array-like of shape (n_grain_sizes, n_x) or (2), default=(100., (0.2, 0.2, 0.3, 0.3))
        Initial thickness of the substratum for each grain size. It can be directly
        an array representing the substratum, or a tuple (thickness, tuple with the
        proportion of each grain size), which is converted into an array during
        initialization.
    event : bool, default=False
        ???
    preinterpolate_curves: bool, default=False
        If True, preinterpolate the sea level and sediment supply curves using
        a cubic interpolation.
    seed : int, default=42
        Controls random number generation for reproductibility.

    Attributes
    ----------
    sequence_ : xarray.Dataset
        Dataset containing the evolution of the elevation and stratigraphy through time.
    record_ : xarray.Dataset
        Dataset containing the final stratigraphy.
        
    References
    ----------
    Storms, J.E.A. (2003) https://doi.org/10.1016/S0025-3227(03)00144-0
    Event-based stratigraphic simulation of wave-dominated shallow-marine environments
    """
    def __init__(self,
                 initial_elevation,
                 sea_level_curve,
                 sediment_supply_curve,
                 spacing=100.,
                 ow_part=0.5,
                 mode=2,
                 W_fw=1.5,
                 W_event=6.,
                 Tidalampl=2,
                 Tide_Acc_VAR=100,
                 TidalSand=0.3,
                 dh_ratio_acc=200000,
                 TDcorr_SF=1.5,
                 TDcorr_OW=1.,
                 erodibility=0.1,
                 BB_max_width=500,
                 A_factor_BB=5,
                 A_factor_SF=10,
                 max_height_SF=1.,
                 max_height_BB=1.,
                 substrate_erosion=1,
                 fallout_rate_bb=0.,
                 fallout_rate_sf=0.0002,
                 texture=(5., 50., 125., 250.),
                 texture_ratio=(0.25, 0.25, 0.25, 0.25),
                 initial_substratum=(100., (0.2, 0.2, 0.3, 0.3)),
                 event=False,
                 preinterpolate_curves=False,
                 seed=42):
        
        self.initial_elevation = np.array(initial_elevation)
        self.sea_level_curve = np.array(sea_level_curve)
        self.sediment_supply_curve = np.array(sediment_supply_curve)
        self.spacing = spacing
        self.ow_part = ow_part
        self.mode = mode
        self.W_fw = W_fw
        self.W_event = W_event
        self.Tidalampl = Tidalampl
        self.Tide_Acc_VAR = Tide_Acc_VAR
        self.TidalSand = TidalSand
        self.dh_ratio_acc = dh_ratio_acc
        self.TDcorr_SF = TDcorr_SF
        self.TDcorr_OW = TDcorr_OW
        self.erodibility = erodibility
        self.BB_max_width = BB_max_width
        self.A_factor_BB = A_factor_BB
        self.A_factor_SF = A_factor_SF
        self.max_height_SF = max_height_SF
        self.max_height_BB = max_height_BB
        self.substrate_erosion = substrate_erosion
        self.fallout_rate_bb = fallout_rate_bb
        self.fallout_rate_sf = fallout_rate_sf
        self.texture = np.array(texture)
        self.texture_ratio = np.array(texture_ratio)
        if (len(initial_substratum) == 2
            and len(initial_substratum[1]) == len(texture)):
            self.initial_substratum = np.full((len(texture), len(initial_elevation)),
                                              initial_substratum[0])
            self.initial_substratum *= np.array(initial_substratum[1])[:, np.newaxis]
        else:
            self.initial_substratum = np.array(initial_substratum)
        self.event = event
        self.preinterpolate_curves = preinterpolate_curves
        self.seed = seed
        
    def _preinterpolate_curves(self, duration, dt_min, dt_fw):
        """
        Preinterpolate the sea level and sediment supply curves using a cubic interpolation
        before using them in Numba.
        TODO: Would be better to have a cubic interpolation in Numba.
        """
        dt_average = (dt_min + dt_fw)//2 - 1 if self.mode == 2 else dt_min
        n_time_steps = int(duration/dt_average) + 1
        
        sea_level_function = interpolate.interp1d(self.sea_level_curve[:, 0],
                                                  self.sea_level_curve[:, 1],
                                                  kind='cubic')
        sea_level_interp = np.empty((n_time_steps, 2))
        sea_level_interp[:, 0] = np.linspace(0., duration, n_time_steps)
        sea_level_interp[:, 1] = sea_level_function(sea_level_interp[:, 0])
        
        sediment_supply_function = interpolate.interp1d(self.sediment_supply_curve[:, 0],
                                                        self.sediment_supply_curve[:, 1],
                                                        kind='cubic')
        sediment_supply_interp = np.empty((n_time_steps, 2))
        sediment_supply_interp[:, 0] = sea_level_interp[:, 0]
        sediment_supply_interp[:, 1] = sediment_supply_function(sediment_supply_interp[:, 0])
        
        return sea_level_interp, sediment_supply_interp
    
    def run(self, duration=10000., dt_min=1., dt_fw=15.):
        """
        Runs BarSim for a given duration.

        Parameters
        ----------
        duration : float, default=10000.
            Duration of the run (in years).
        dt_min : float, default=10000.
            Minimum time step (in years).
        dt_fw : float, default=10000.
            ??? (in years).
        """
        if self.preinterpolate_curves == True:
            sea_level_curve, sediment_supply_curve = self._preinterpolate_curves(duration, dt_min, dt_fw)
        else:
            sea_level_curve, sediment_supply_curve = self.sea_level_curve, self.sediment_supply_curve
        time, sea_level, sediment_supply, elevation, stratigraphy, facies = _run(self.initial_elevation, self.initial_substratum,
                                                                                 sea_level_curve, sediment_supply_curve,
                                                                                 duration, self.spacing, self.ow_part, self.mode,
                                                                                 self.W_fw, self.W_event, dt_min, dt_fw,
                                                                                 self.Tidalampl, self.Tide_Acc_VAR,
                                                                                 self.TidalSand, self.dh_ratio_acc,
                                                                                 self.TDcorr_SF, self.TDcorr_OW,
                                                                                 self.erodibility, self.BB_max_width,
                                                                                 self.A_factor_BB, self.A_factor_SF,
                                                                                 self.max_height_SF, self.max_height_BB,
                                                                                 self.substrate_erosion, self.fallout_rate_bb,
                                                                                 self.fallout_rate_sf, self.texture,
                                                                                 self.texture_ratio, self.event, self.seed)

        self.sequence_ = xr.Dataset(
            data_vars={
                'Sea level': (('Time',), sea_level, {'units': 'meter', 'description': 'sea level through time'}),
                'Sediment supply': (('Time',), sediment_supply, {'units': 'cubic meter', 'description': 'sediment supply through time'}),
                'Elevation': (('Time', 'X'), elevation, {'units': 'meter', 'description': 'elevation through time'}),
                'Stratigraphy': (('Grain size', 'Time', 'X'), stratigraphy, {'units': 'meter', 'description': 'deposit thickness through time'}),
                'Facies': (('Time', 'X'), facies, {'units': '', 'description': 'facies through time: 1. substratum, 2. coastal plain, 3. lagoon, 4. barrier island, 5. upper shoreface, 6. lower shoreface, 7. offshore'}),
            },
            coords={
                'X': np.linspace(self.spacing/2.,
                                 self.spacing*(elevation.shape[1] - 0.5),
                                 elevation.shape[1]),
                'Time': time,
                'Grain size': np.array(self.texture),
            },
        )
        self.sequence_['X'].attrs['units'] = 'meter'
        self.sequence_['Time'].attrs['units'] = 'year'
        self.sequence_['Grain size'].attrs['units'] = 'micrometer'
        
    def regrid(self, z_min, z_max, z_step):
        """
        Interpolates the time stratigraphy into a regular grid.

        Parameters
        ----------
        z_min : float
            Lower border of the grid.
        z_max : float
            Upper border of the grid.
        z_step : float
            Resolution of the grid.
        """
        stratigraphy, facies = _regrid_stratigraphy(self.sequence_['Elevation'][-1].to_numpy(),
                                                    self.sequence_['Stratigraphy'].to_numpy(),
                                                    self.sequence_['Facies'].to_numpy(),
                                                    z_min, z_max, z_step)
        median = _compute_median_grain_size(stratigraphy, self.texture)
        
        self.record_ = xr.Dataset(
            data_vars={
                'Median grain size': (('Z', 'X'), median, {'units': 'micrometer', 'description': 'median grain size'}),
                'Major facies': (('Z', 'X'), np.argmax(facies, axis=0), {'units': '', 'description': 'major facies: 0. none, 1. substratum, 2. coastal plain, 3. lagoon, 4. barrier island, 5. upper shoreface, 6. lower shoreface, 7. offshore'}),
                'Stratigraphy': (('Grain size', 'Z', 'X'), stratigraphy/z_step, {'units': 'meter', 'description': 'fraction of each grain size in a cell'}),
                'Facies': (('Environment', 'Z', 'X'), facies, {'units': '', 'description': 'fraction of each facies in a cell'}),
            },
            coords={
                'X': np.linspace(self.spacing/2.,
                                 self.spacing*(median.shape[1] - 0.5),
                                 median.shape[1]),
                'Z': np.linspace(z_min + z_step/2.,
                                 z_max - z_step/2.,
                                 median.shape[0]),
                'Grain size': self.texture,
                'Environment': ['none', 'substratum', 'coastal plain', 'lagoon', 'barrier island', 'upper shoreface', 'lower shoreface', 'offshore'],
            },
        )
        self.record_['X'].attrs['units'] = 'meter'
        self.record_['Z'].attrs['units'] = 'meter'
        self.record_['Grain size'].attrs['units'] = 'micrometer'


################################################################################
# BarSim 2.5D

@nb.jit(nopython=True, parallel=True)
def _run_multiple(initial_elevation, initial_substratum, sea_level_curve, sediment_supply_curve,
                  duration, spacing, ow_part, mode, W_fw, W_event, dt_min, dt_fw, Tidalampl,
                  Tide_Acc_VAR, TidalSand, dh_ratio_acc, TDcorr_SF, TDcorr_OW,
                  erodibility, BB_max_width, A_factor_BB, A_factor_SF, max_height_SF,
                  max_height_BB, substrate_erosion, fallout_rate_bb, fallout_rate_sf,
                  texture, texture_ratio, event, z_min, z_max, z_step, seed):
    
    stratigraphy = np.empty((len(texture),
                             int((z_max - z_min)/z_step),
                             initial_elevation.shape[0],
                             initial_elevation.shape[1]))
    facies = np.empty((8,
                       int((z_max - z_min)/z_step),
                       initial_elevation.shape[0],
                       initial_elevation.shape[1]))
    median = np.empty((int((z_max - z_min)/z_step),
                       initial_elevation.shape[0],
                       initial_elevation.shape[1]))
    major_facies = np.empty((int((z_max - z_min)/z_step),
                             initial_elevation.shape[0],
                             initial_elevation.shape[1]))
    for i in nb.prange(initial_elevation.shape[0]):
        _, _, _, elevation, time_stratigraphy, time_facies = _run(initial_elevation[i], initial_substratum,
                                                                  sea_level_curve, sediment_supply_curve[i],
                                                                  duration, spacing, ow_part, mode,
                                                                  W_fw, W_event, dt_min, dt_fw,
                                                                  Tidalampl, Tide_Acc_VAR,
                                                                  TidalSand, dh_ratio_acc,
                                                                  TDcorr_SF, TDcorr_OW,
                                                                  erodibility, BB_max_width,
                                                                  A_factor_BB, A_factor_SF,
                                                                  max_height_SF, max_height_BB,
                                                                  substrate_erosion, fallout_rate_bb,
                                                                  fallout_rate_sf, texture,
                                                                  texture_ratio, event, seed)
        stratigraphy[..., i], facies[..., i] = _regrid_stratigraphy(elevation[-1], time_stratigraphy,
                                                                    time_facies, z_min, z_max, z_step)
        median[..., i] = _compute_median_grain_size(stratigraphy[..., i], texture)
        
    return stratigraphy, facies, median


class BarSimPseudo3D:
    """
    2.5D implementation of the multiple grain-size, process-response model BarSim.
    
    TODO: Rename input parameters, pick good defaults, reorder, and fill the doc.

    Parameters
    ----------
    initial_elevation : array-like of shape (n_y, n_x)
        The initial elevation.
    sea_level_curve : array-like of shape (n_t, 2)
        The inflexion points of sea level in time.
    sediment_supply_curve : array-like of shape (n_x, n_t, 2)
        The inflexion points of sediment supply in time.
    spacing : array-like of shape (2,), default=(100., 100.).
        The grid resolution.
    ow_part : float, default=0.5
        ???
    mode : int, default=2
        ???
    W_fw : float, default=1.5
        ???
    W_event : float, default=6.
        ???
    Tidalampl : ???, default=2
        ???
    Tide_Acc_VAR : ???, default=100
        ???
    TidalSand : float, default=0.3
        ???
    dh_ratio_acc : ???, default=200000
        ???
    TDcorr_SF : float, default=1.5
        ???
    TDcorr_OW : float, default=1.
        ???
    erodibility : float, default=0.1
        ???
    BB_max_width : ???, default=500
        ???
    A_factor_BB : ???, default=5
        ???
    A_factor_SF : ???, default=10
        ???
    A_factor_SF : ???, default=10
        ???
    max_height_SF : float, default=1.
        ???
    max_height_BB : float, default=1.
        ???
    substrate_erosion : ???, default=1
        ??? That parameter is never used
    fallout_rate_bb : float, default=0.
        ???
    fallout_rate_sf : float, default=0.0002
        ???
    texture : array-like of shape (n_grain_sizes,), default=(5, 50, 125, 250)
        ???
    texture_ratio : array-like of shape (n_grain_sizes,), default=(0.25, 0.25, 0.25, 0.25)
        ???
    initial_substratum : array-like of shape (n_grain_sizes, n_x) or (2), default=(100., (0.2, 0.2, 0.3, 0.3))
        Initial thickness of the substratum for each grain size. It can be directly
        an array representing the substratum, or a tuple (thickness, tuple with the
        proportion of each grain size), which is converted into an array during
        initialization.
    event : bool, default=False
        ???
    preinterpolate_curves: bool, default=False
        If True, preinterpolate the sea level and sediment supply curves using
        a cubic interpolation.
    seed : int, default=42
        Controls random number generation for reproductibility.

    Attributes
    ----------
    sequence_ : xarray.Dataset
        Dataset containing the evolution of the elevation and stratigraphy through time.
        
    References
    ----------
    Storms, J.E.A. (2003) https://doi.org/10.1016/S0025-3227(03)00144-0
    Event-based stratigraphic simulation of wave-dominated shallow-marine environments
    """
    def __init__(self,
                 initial_elevation,
                 sea_level_curve,
                 sediment_supply_curve,
                 spacing=(100., 100.),
                 ow_part=0.5,
                 mode=2,
                 W_fw=1.5,
                 W_event=6.,
                 Tidalampl=2,
                 Tide_Acc_VAR=100,
                 TidalSand=0.3,
                 dh_ratio_acc=200000,
                 TDcorr_SF=1.5,
                 TDcorr_OW=1.,
                 erodibility=0.1,
                 BB_max_width=500,
                 A_factor_BB=5,
                 A_factor_SF=10,
                 max_height_SF=1.,
                 max_height_BB=1.,
                 substrate_erosion=1,
                 fallout_rate_bb=0.,
                 fallout_rate_sf=0.0002,
                 texture=(5., 50., 125., 250.),
                 texture_ratio=(0.25, 0.25, 0.25, 0.25),
                 initial_substratum=(100., (0.2, 0.2, 0.3, 0.3)),
                 event=False,
                 preinterpolate_curves=False,
                 n_jobs=-1,
                 seed=42):
        
        self.initial_elevation = np.array(initial_elevation)
        self.sea_level_curve = np.array(sea_level_curve)
        self.sediment_supply_curve = np.array(sediment_supply_curve)
        self.spacing = spacing
        self.ow_part = ow_part
        self.mode = mode
        self.W_fw = W_fw
        self.W_event = W_event
        self.Tidalampl = Tidalampl
        self.Tide_Acc_VAR = Tide_Acc_VAR
        self.TidalSand = TidalSand
        self.dh_ratio_acc = dh_ratio_acc
        self.TDcorr_SF = TDcorr_SF
        self.TDcorr_OW = TDcorr_OW
        self.erodibility = erodibility
        self.BB_max_width = BB_max_width
        self.A_factor_BB = A_factor_BB
        self.A_factor_SF = A_factor_SF
        self.max_height_SF = max_height_SF
        self.max_height_BB = max_height_BB
        self.substrate_erosion = substrate_erosion
        self.fallout_rate_bb = fallout_rate_bb
        self.fallout_rate_sf = fallout_rate_sf
        self.texture = np.array(texture)
        self.texture_ratio = np.array(texture_ratio)
        if (len(initial_substratum) == 2
            and len(initial_substratum[1]) == len(texture)):
            self.initial_substratum = np.full((len(texture), len(initial_elevation)),
                                              initial_substratum[0])
            self.initial_substratum *= np.array(initial_substratum[1])[:, np.newaxis]
        else:
            self.initial_substratum = np.array(initial_substratum)
        self.event = event
        self.preinterpolate_curves = preinterpolate_curves
        self.n_jobs = n_jobs
        self.seed = seed
        
    def _preinterpolate_curves(self, duration, dt_min, dt_fw):
        """
        Preinterpolate the sea level and sediment supply curves using a cubic interpolation
        before using them in Numba.
        TODO: Would be better to have a cubic interpolation in Numba.
        """
        dt_average = (dt_min + dt_fw)//2 - 1 if self.mode == 2 else dt_min
        n_time_steps = int(duration/dt_average) + 1
        
        sea_level_function = interpolate.interp1d(self.sea_level_curve[:, 0],
                                                  self.sea_level_curve[:, 1],
                                                  kind='cubic')
        sea_level_interp = np.empty((n_time_steps, 2))
        sea_level_interp[:, 0] = np.linspace(0., duration, n_time_steps)
        sea_level_interp[:, 1] = sea_level_function(sea_level_interp[:, 0])
        
        sediment_supply_interp = np.empty((len(self.sediment_supply_curve), n_time_steps, 2))
        for i in range(len(self.sediment_supply_curve)):
            sediment_supply_function = interpolate.interp1d(self.sediment_supply_curve[i, :, 0],
                                                            self.sediment_supply_curve[i, :, 1],
                                                            kind='cubic')
            sediment_supply_interp[i, :, 0] = sea_level_interp[:, 0]
            sediment_supply_interp[i, :, 1] = sediment_supply_function(sediment_supply_interp[i, :, 0])
        
        return sea_level_interp, sediment_supply_interp
    
    def run(self, z_min, z_max, z_step, duration=10000., dt_min=1., dt_fw=15.):
        """
        Runs BarSim for a given duration.

        Parameters
        ----------
        duration : float, default=10000.
            Duration of the run (in years).
        dt_min : float, default=10000.
            Minimum time step (in years).
        dt_fw : float, default=10000.
            ??? (in years).
        """
        if self.preinterpolate_curves == True:
            sea_level_curve, sediment_supply_curve = self._preinterpolate_curves(duration, dt_min, dt_fw)
        else:
            sea_level_curve, sediment_supply_curve = self.sea_level_curve, self.sediment_supply_curve
        stratigraphy, facies, median = _run_multiple(self.initial_elevation, self.initial_substratum, sea_level_curve,
                                                     sediment_supply_curve, duration, self.spacing[1], self.ow_part,
                                                     self.mode, self.W_fw, self.W_event, dt_min, dt_fw, self.Tidalampl,
                                                     self.Tide_Acc_VAR, self.TidalSand, self.dh_ratio_acc, self.TDcorr_SF,
                                                     self.TDcorr_OW, self.erodibility, self.BB_max_width, self.A_factor_BB,
                                                     self.A_factor_SF, self.max_height_SF, self.max_height_BB, self.substrate_erosion,
                                                     self.fallout_rate_bb, self.fallout_rate_sf, self.texture, self.texture_ratio,
                                                     self.event, z_min, z_max, z_step, self.seed)

        self.record_ = xr.Dataset(
            data_vars={
                'Median grain size': (('Z', 'Y', 'X'), median, {'units': 'micrometer', 'description': 'median grain size'}),
                'Major facies': (('Z', 'Y', 'X'), np.argmax(facies, axis=0), {'units': '', 'description': 'major facies: 0. none, 1. substratum, 2. coastal plain, 3. lagoon, 4. barrier island, 5. upper shoreface, 6. lower shoreface, 7. offshore'}),
                'Stratigraphy': (('Grain size', 'Z', 'Y', 'X'), stratigraphy/z_step, {'units': 'meter', 'description': 'fraction of each grain size in a cell'}),
                'Facies': (('Environment', 'Z', 'Y', 'X'), facies, {'units': '', 'description': 'fraction of each facies in a cell'}),
            },
            coords={
                'X': np.linspace(self.spacing[1]/2.,
                                 self.spacing[1]*(median.shape[2] - 0.5),
                                 median.shape[2]),
                'Y': np.linspace(self.spacing[0]/2.,
                                 self.spacing[0]*(median.shape[1] - 0.5),
                                 median.shape[1]),
                'Z': np.linspace(z_min + z_step/2.,
                                 z_max - z_step/2.,
                                 median.shape[0]),
                'Grain size': self.texture,
                'Environment': ['none', 'substratum', 'coastal plain', 'lagoon', 'barrier island', 'upper shoreface', 'lower shoreface', 'offshore'],
            },
        )
        self.record_['X'].attrs['units'] = 'meter'
        self.record_['Y'].attrs['units'] = 'meter'
        self.record_['Z'].attrs['units'] = 'meter'
        self.record_['Grain size'].attrs['units'] = 'micrometer'
        
    def mesh(self, zscale=1.):
        """
        Turns record_ into a PyVista mesh object.
        
        Parameters
        ----------
        zscale : float, default=1.
            Vertical scale factor.
        """
        x = self.record_['X'].to_numpy().copy()
        x -= (x[1] - x[0])/2
        x = np.append(x, x[-1] + (x[1] - x[0]))
        y = self.record_['Y'].to_numpy().copy()
        y -= (y[1] - y[0])/2
        y = np.append(y, y[-1] + (y[1] - y[0]))
        z = self.record_['Z'].to_numpy().copy()
        z -= (z[1] - z[0])/2
        z = np.append(z, z[-1] + (z[1] - z[0]))
        xx, yy, zz = np.meshgrid(x, y, zscale*z)
        mesh = pv.StructuredGrid(xx, yy, zz)
        median = self.record_['Median grain size'].to_numpy().copy()
        median[(self.record_['Facies'][0] == 1) | (self.record_['Facies'][1] == 1)] = np.nan
        mesh['Median grain size'] = median.ravel()
        major_facies = self.record_['Major facies'].to_numpy().astype(float)
        major_facies[(self.record_['Facies'][0] > 0) | (self.record_['Facies'][1] > 0)] = np.nan
        mesh['Major facies'] = major_facies.ravel()
        
        return mesh
