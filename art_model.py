"""
Asymptotic Radiative Transfer (ART) Model for Snow and Ice
Based on Kokhanovsky & Zege (2004) "Scattering optics of snow"

This module provides analytical solutions for snow/ice radiative transfer
used for generating lookup tables in ISSIA processing.

References:
-----------
Kokhanovsky, A.A. and Zege, E.P., 2004. Scattering optics of snow. 
    Applied optics, 43(7), pp.1589-1602.

Zege, E.P., Ivanov, A.P. and Katsev, I.L., 1991. Image transfer through 
    a scattering medium. Springer Science & Business Media.

Libois, Q., Picard, G., France, J.L., Arnaud, L., Dumont, M., Carmagnola, C.M. 
    and King, M.D., 2013. Influence of grain shape on light penetration in snow. 
    The Cryosphere, 7(6), pp.1803-1818.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import interp1d
import warnings


class ARTSnowModel:
    """
    Asymptotic Radiative Transfer model for clean and polluted snow
    
    This implementation provides analytical solutions for:
    - Spectral albedo (directional and hemispherical)
    - Bidirectional reflectance distribution function (BRDF)
    - Effects of grain size, shape, and light-absorbing particles
    """
    
    def __init__(self, wavelengths: np.ndarray):
        """
        Initialize ART snow model
        
        Parameters:
        -----------
        wavelengths : np.ndarray
            Wavelength array in nanometers (nm)
        """
        self.wavelengths = wavelengths
        self.wl_um = wavelengths / 1000.0  # Convert to micrometers
        
        # Load ice optical properties
        self.n_ice = self._get_ice_refractive_index()
        self.k_ice = self._get_ice_absorption_index()
        
    def _get_ice_refractive_index(self) -> np.ndarray:
        """
        Get ice refractive index (real part) as function of wavelength
        Based on Warren & Brandt (2008)
        
        Returns:
        --------
        n_ice : np.ndarray
            Real refractive index
        """
        # Simplified parameterization - in production use Warren & Brandt (2008) table
        # Refractive index is nearly constant (~1.31) in visible/NIR
        n_ice = np.ones_like(self.wl_um) * 1.31
        
        # Small wavelength dependence
        n_ice += 0.001 * (self.wl_um - 0.5)
        
        return n_ice
    
    def _get_ice_absorption_index(self) -> np.ndarray:
        """
        Get ice absorption index (imaginary part) as function of wavelength
        Based on Warren & Brandt (2008) and Picard et al. (2016)
        
        Returns:
        --------
        k_ice : np.ndarray
            Imaginary refractive index (absorption)
        """
        # This is a simplified parameterization
        # For production, use actual Warren & Brandt (2008) data
        
        wl = self.wl_um
        k_ice = np.zeros_like(wl)
        
        # Visible (very low absorption)
        vis_mask = wl < 0.7
        k_ice[vis_mask] = 1e-9 * np.exp((wl[vis_mask] - 0.4) * 3)
        
        # NIR (moderate absorption, 1030 nm feature)
        nir_mask = (wl >= 0.7) & (wl < 1.4)
        # Ice absorption feature around 1030 nm
        k_ice[nir_mask] = 1e-7 * np.exp((wl[nir_mask] - 0.7) * 8)
        
        # Add 1030 nm absorption peak
        peak_1030 = np.exp(-((wl - 1.030)**2) / (2 * 0.05**2))
        k_ice += peak_1030 * 2e-6
        
        # SWIR (strong absorption)
        swir_mask = wl >= 1.4
        k_ice[swir_mask] = 1e-6 * np.exp((wl[swir_mask] - 1.4) * 3)
        
        return k_ice
    
    def calculate_absorption_coefficient(self, wavelength_um: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate ice absorption coefficient
        
        Parameters:
        -----------
        wavelength_um : np.ndarray, optional
            Wavelengths in micrometers (uses self.wl_um if None)
            
        Returns:
        --------
        alpha_abs : np.ndarray
            Absorption coefficient (mm^-1)
        """
        if wavelength_um is None:
            wavelength_um = self.wl_um
            k_ice = self.k_ice
        else:
            # Interpolate
            interp_func = interp1d(self.wl_um, self.k_ice, 
                                  bounds_error=False, fill_value='extrapolate')
            k_ice = interp_func(wavelength_um)
        
        # Absorption coefficient: α = 4πk/λ
        alpha_abs = (4 * np.pi * k_ice) / wavelength_um
        
        return alpha_abs  # mm^-1
    
    def calculate_scattering_coefficient(self, 
                                        grain_size_um: float,
                                        shape_parameter: float = 1.8) -> float:
        """
        Calculate scattering cross section per unit volume
        
        Parameters:
        -----------
        grain_size_um : float
            Effective grain radius (micrometers)
        shape_parameter : float
            Shape factor (B parameter in Kokhanovsky & Zege)
            1.25 for spheres, ~1.8 for more realistic shapes
            
        Returns:
        --------
        sigma : float
            Scattering coefficient (mm^-1)
        """
        # Convert grain size to mm
        grain_mm = grain_size_um / 1000.0
        
        # Scattering coefficient = 3/(2*r_eff) for spheres
        # Modified by shape parameter
        sigma = (3.0 / (2.0 * grain_mm)) * shape_parameter
        
        return sigma  # mm^-1
    
    def calculate_plane_albedo(self,
                              grain_size_um: float,
                              solar_zenith_deg: float,
                              impurity_absorption: Optional[np.ndarray] = None,
                              shape_parameter: float = 1.8,
                              asymmetry_parameter: float = 0.89) -> np.ndarray:
        """
        Calculate plane albedo (hemispherical reflectance) for direct illumination
        Using ART analytical solution
        
        Parameters:
        -----------
        grain_size_um : float
            Effective grain radius (micrometers)
        solar_zenith_deg : float
            Solar zenith angle (degrees)
        impurity_absorption : np.ndarray, optional
            Additional absorption by impurities (mm^-1)
        shape_parameter : float
            Grain shape parameter (B in ART theory)
        asymmetry_parameter : float
            Phase function asymmetry parameter (g)
            
        Returns:
        --------
        albedo : np.ndarray
            Spectral plane albedo
        """
        # Convert angle
        mu0 = np.cos(np.deg2rad(solar_zenith_deg))
        
        # Ice absorption
        alpha_ice = self.calculate_absorption_coefficient()
        
        # Total absorption (ice + impurities)
        if impurity_absorption is not None:
            alpha_total = alpha_ice + impurity_absorption
        else:
            alpha_total = alpha_ice
        
        # Scattering coefficient
        sigma = self.calculate_scattering_coefficient(grain_size_um, shape_parameter)
        
        # Asymptotic flux extinction coefficient (Kokhanovsky & Zege Eq. 9)
        # k_e = sqrt(3 * alpha_total * sigma * (1-g))
        u = np.sqrt(3.0 * (1.0 - asymmetry_parameter))
        k_e = np.sqrt(alpha_total * sigma) * u
        
        # Escape function R0 (Kokhanovsky & Zege Eq. 12)
        # For conservative scattering
        R0 = self._calculate_escape_function(mu0)
        
        # Plane albedo (Kokhanovsky & Zege Eq. 20)
        # α = exp(-4 * μ0 * k_e) * R0
        albedo = np.exp(-4.0 * mu0 * k_e) * R0
        
        # Ensure physically valid range
        albedo = np.clip(albedo, 0, 1)
        
        return albedo
    
    def calculate_spherical_albedo(self,
                                   grain_size_um: float,
                                   impurity_absorption: Optional[np.ndarray] = None,
                                   shape_parameter: float = 1.8,
                                   asymmetry_parameter: float = 0.89) -> np.ndarray:
        """
        Calculate spherical albedo (bi-hemispherical reflectance) for diffuse illumination
        
        Parameters:
        -----------
        grain_size_um : float
            Effective grain radius (micrometers)
        impurity_absorption : np.ndarray, optional
            Additional absorption by impurities (mm^-1)
        shape_parameter : float
            Grain shape parameter
        asymmetry_parameter : float
            Phase function asymmetry parameter
            
        Returns:
        --------
        albedo : np.ndarray
            Spectral spherical (white-sky) albedo
        """
        # Ice absorption
        alpha_ice = self.calculate_absorption_coefficient()
        
        # Total absorption
        if impurity_absorption is not None:
            alpha_total = alpha_ice + impurity_absorption
        else:
            alpha_total = alpha_ice
        
        # Scattering coefficient
        sigma = self.calculate_scattering_coefficient(grain_size_um, shape_parameter)
        
        # Asymptotic flux extinction coefficient
        u = np.sqrt(3.0 * (1.0 - asymmetry_parameter))
        k_e = np.sqrt(alpha_total * sigma) * u
        
        # Spherical albedo (Kokhanovsky & Zege Eq. 21)
        # α_sph = exp(-u_sph * k_e)
        # where u_sph ≈ 4 for snow (from numerical integration)
        u_sph = 4.0
        
        albedo = np.exp(-u_sph * k_e)
        
        # Ensure valid range
        albedo = np.clip(albedo, 0, 1)
        
        return albedo
    
    def calculate_brdf(self,
                      grain_size_um: float,
                      solar_zenith_deg: float,
                      view_zenith_deg: float,
                      relative_azimuth_deg: float,
                      impurity_absorption: Optional[np.ndarray] = None,
                      shape_parameter: float = 1.8,
                      asymmetry_parameter: float = 0.89) -> np.ndarray:
        """
        Calculate BRDF (bidirectional reflectance distribution function)
        
        This includes both the diffuse component and angular dependence
        
        Parameters:
        -----------
        grain_size_um : float
            Effective grain radius (micrometers)
        solar_zenith_deg : float
            Solar zenith angle (degrees)
        view_zenith_deg : float
            Viewing zenith angle (degrees)
        relative_azimuth_deg : float
            Relative azimuth angle (degrees)
        impurity_absorption : np.ndarray, optional
            Additional absorption by impurities
        shape_parameter : float
            Grain shape parameter
        asymmetry_parameter : float
            Phase function asymmetry parameter
            
        Returns:
        --------
        brdf : np.ndarray
            Spectral BRDF
        """
        # Calculate base albedo
        plane_albedo = self.calculate_plane_albedo(
            grain_size_um, solar_zenith_deg, 
            impurity_absorption, shape_parameter, asymmetry_parameter
        )
        
        # Angular correction factors
        mu_i = np.cos(np.deg2rad(solar_zenith_deg))
        mu_v = np.cos(np.deg2rad(view_zenith_deg))
        phi = np.deg2rad(relative_azimuth_deg)
        
        # Scattering angle (cosine)
        cos_theta = mu_i * mu_v + \
                    np.sqrt(1 - mu_i**2) * np.sqrt(1 - mu_v**2) * np.cos(phi)
        
        # Phase function (Henyey-Greenstein)
        g = asymmetry_parameter
        p_hg = (1 - g**2) / (1 + g**2 - 2*g*cos_theta)**1.5
        
        # Normalization factor (4π for BRDF)
        p_hg_norm = p_hg / (4 * np.pi)
        
        # BRDF includes angular dependence and viewing geometry
        # Simple approximation: BRDF ≈ albedo * phase_function * geometry_factor
        geometry_factor = (mu_i + mu_v) / (mu_i * mu_v)
        
        brdf = plane_albedo * p_hg_norm * geometry_factor
        
        # Ensure reasonable values
        brdf = np.clip(brdf, 0, 2)  # BRDF can be > 1
        
        return brdf
    
    def _calculate_escape_function(self, mu: float) -> float:
        """
        Calculate escape function R0 for conservative scattering
        
        Parameters:
        -----------
        mu : float
            Cosine of angle
            
        Returns:
        --------
        R0 : float
            Escape function value
        """
        # Escape function for conservative scattering
        # Approximation from Kokhanovsky & Zege
        R0 = (mu + 1.5 * mu * (1 - mu)) / (mu + 1.5)
        
        return R0
    
    def add_impurities(self,
                      impurity_type: str = 'soot',
                      mass_concentration: float = 100.0) -> np.ndarray:
        """
        Calculate additional absorption from light-absorbing particles
        
        Parameters:
        -----------
        impurity_type : str
            Type of impurity ('soot', 'dust', 'algae')
        mass_concentration : float
            Mass concentration (ng/g or ppb)
            
        Returns:
        --------
        alpha_imp : np.ndarray
            Additional absorption coefficient (mm^-1)
        """
        # Mass absorption cross-sections (m²/g) from literature
        # These are spectral and depend on impurity type
        
        if impurity_type == 'soot':
            # Black carbon - relatively spectrally flat
            mac = 6.0  # m²/g at 550 nm
            spectral_dependence = (0.55 / self.wl_um)**1.0
            
        elif impurity_type == 'dust':
            # Mineral dust - stronger wavelength dependence
            mac = 0.5  # m²/g at 550 nm
            spectral_dependence = (0.55 / self.wl_um)**3.0
            
        elif impurity_type == 'algae':
            # Glacier algae - absorption peaks in visible
            mac = 1.0  # m²/g at 550 nm
            # Algae have chlorophyll absorption
            chl_absorption = np.exp(-((self.wl_um - 0.680)**2) / (2 * 0.05**2))
            spectral_dependence = (0.55 / self.wl_um)**2.0 + chl_absorption * 2
            
        else:
            raise ValueError(f"Unknown impurity type: {impurity_type}")
        
        # Calculate absorption coefficient
        # α_imp = MAC * C / ρ_ice
        # where C is concentration (g/g) and ρ_ice is ice density (0.917 g/cm³)
        
        rho_ice = 917  # kg/m³ = 0.917 g/cm³
        conc_g_per_g = mass_concentration * 1e-9  # ng/g to g/g
        
        # MAC in m²/g, need mm^-1
        # α_imp (mm^-1) = MAC (m²/g) * C (g/g) * 1e6 (m² to mm²) / density_factor
        alpha_imp = (mac * spectral_dependence * conc_g_per_g * 1000) / rho_ice
        
        return alpha_imp


# Convenience functions for ISSIA integration
def simulate_snow_reflectance(wavelengths: np.ndarray,
                             grain_size_um: float,
                             solar_zenith: float,
                             view_zenith: float,
                             relative_azimuth: float,
                             impurity_type: Optional[str] = None,
                             impurity_conc: float = 0.0) -> np.ndarray:
    """
    Simulate snow reflectance spectrum using ART model
    
    Parameters:
    -----------
    wavelengths : np.ndarray
        Wavelengths (nm)
    grain_size_um : float
        Grain size (micrometers)
    solar_zenith : float
        Solar zenith angle (degrees)
    view_zenith : float
        View zenith angle (degrees)
    relative_azimuth : float
        Relative azimuth (degrees)
    impurity_type : str, optional
        Type of impurity
    impurity_conc : float
        Impurity concentration (ng/g)
        
    Returns:
    --------
    reflectance : np.ndarray
        Spectral reflectance
    """
    model = ARTSnowModel(wavelengths)
    
    # Add impurities if specified
    if impurity_type and impurity_conc > 0:
        impurity_abs = model.add_impurities(impurity_type, impurity_conc)
    else:
        impurity_abs = None
    
    # Calculate BRDF
    brdf = model.calculate_brdf(
        grain_size_um,
        solar_zenith,
        view_zenith,
        relative_azimuth,
        impurity_abs
    )
    
    return brdf


def simulate_snow_albedo(wavelengths: np.ndarray,
                        grain_size_um: float,
                        impurity_type: Optional[str] = None,
                        impurity_conc: float = 0.0) -> np.ndarray:
    """
    Simulate snow white-sky (hemispherical) albedo using ART model
    
    Parameters:
    -----------
    wavelengths : np.ndarray
        Wavelengths (nm)
    grain_size_um : float
        Grain size (micrometers)
    impurity_type : str, optional
        Type of impurity
    impurity_conc : float
        Impurity concentration (ng/g)
        
    Returns:
    --------
    albedo : np.ndarray
        Spectral albedo
    """
    model = ARTSnowModel(wavelengths)
    
    # Add impurities if specified
    if impurity_type and impurity_conc > 0:
        impurity_abs = model.add_impurities(impurity_type, impurity_conc)
    else:
        impurity_abs = None
    
    # Calculate spherical albedo
    albedo = model.calculate_spherical_albedo(
        grain_size_um,
        impurity_abs
    )
    
    return albedo


if __name__ == "__main__":
    """
    Example usage and testing
    """
    import matplotlib.pyplot as plt
    
    # Test wavelengths
    wavelengths = np.linspace(380, 1400, 200)
    
    # Initialize model
    model = ARTSnowModel(wavelengths)
    
    # Test different grain sizes
    grain_sizes = [50, 100, 200, 500, 1000]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot plane albedo for different grain sizes
    for gs in grain_sizes:
        albedo = model.calculate_plane_albedo(gs, solar_zenith_deg=30)
        ax1.plot(wavelengths, albedo, label=f'{gs} μm', linewidth=2)
    
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Plane Albedo', fontsize=12)
    ax1.set_title('Snow Albedo vs Grain Size (SZA=30°)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([380, 1400])
    ax1.axvline(1030, color='red', linestyle='--', alpha=0.5, label='1030 nm')
    
    # Plot effect of impurities
    clean = model.calculate_plane_albedo(200, solar_zenith_deg=30)
    
    soot_100 = model.calculate_plane_albedo(
        200, 30, model.add_impurities('soot', 100)
    )
    soot_1000 = model.calculate_plane_albedo(
        200, 30, model.add_impurities('soot', 1000)
    )
    
    ax2.plot(wavelengths, clean, 'b-', linewidth=2, label='Clean')
    ax2.plot(wavelengths, soot_100, 'g-', linewidth=2, label='100 ng/g soot')
    ax2.plot(wavelengths, soot_1000, 'r-', linewidth=2, label='1000 ng/g soot')
    
    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Plane Albedo', fontsize=12)
    ax2.set_title('Effect of Black Carbon (200 μm grain, SZA=30°)', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([380, 1400])
    
    plt.tight_layout()
    plt.savefig('art_snow_model_test.png', dpi=150, bbox_inches='tight')
    print("Saved test plot to: art_snow_model_test.png")
    
    # Print some statistics
    print("\nGrain Size vs 1030nm Band Depth:")
    print("-" * 40)
    for gs in grain_sizes:
        albedo = model.calculate_plane_albedo(gs, 30)
        idx_1030 = np.argmin(np.abs(wavelengths - 1030))
        idx_950 = np.argmin(np.abs(wavelengths - 950))
        idx_1100 = np.argmin(np.abs(wavelengths - 1100))
        
        # Continuum removal
        continuum = np.interp([wavelengths[idx_1030]], 
                             [wavelengths[idx_950], wavelengths[idx_1100]],
                             [albedo[idx_950], albedo[idx_1100]])[0]
        
        band_depth = 1 - albedo[idx_1030]/continuum
        print(f"{gs:4d} μm: Band depth = {band_depth:.4f}")
