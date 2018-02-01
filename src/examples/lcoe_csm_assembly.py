"""
LCOE_csm_ssembly.py
Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.api import IndepVarComp, Component, Problem, Group
from fused_wind import create_interface , FUSED_Object , FUSED_OpenMDAO , set_output, set_input, fusedvar

# NREL cost and scaling model sub-assemblies
from nrel_csm_tcc import tcc_csm_fused
from nrel_csm_bos import bos_csm_fused
from nrel_csm_opex  import opex_csm_fused
from nrel_csm_fin import fin_csm_fused
from nrel_csm_aep import aep_csm_fused

import numpy as np

def example():

    # openmdao example of execution
    root = Group()
    root.add('desvars',IndepVarComp([('machine_rating',5000.0),
    																 ('rotor_diameter', 126.0),
    																 ('hub_height', 90.0),
    																 ('turbine_number', 100.0),
    																 ('year', 2009.0),
    																 ('month',12.0),
    																 ]),promotes=['*'])
    root.add('bos_csm_test', FUSED_OpenMDAO(bos_csm_fused()), promotes=['*'])
    root.add('tcc_csm_test', FUSED_OpenMDAO(tcc_csm_fused()), promotes=['*'])
    root.add('fin_csm_test', FUSED_OpenMDAO(fin_csm_fused(fixed_charge_rate = 0.12, construction_finance_rate=0.0, tax_rate = 0.4, discount_rate = 0.07, \
                      construction_time = 1.0, project_lifetime = 20.0, sea_depth = 20.0)), promotes=['*'])
    root.add('bos_opex_test', FUSED_OpenMDAO(opex_csm_fused()), promotes=['*'])
    root.add('aep_test', FUSED_OpenMDAO(aep_csm_fused()), promotes=['*'])
    prob = Problem(root)
    prob.setup()


    # set inputs
    # simple test of module
    # Turbine inputs
    prob['rotor_diameter'] = 126.0
    prob['blade_number'] = 3
    prob['hub_height'] = 90.0    
    prob['machine_rating'] = 5000.0

    # Rotor force calculations for nacelle inputs
    maxTipSpd = 80.0
    maxEfficiency = 0.90201
    ratedWindSpd = 11.5064
    thrustCoeff = 0.50
    airDensity = 1.225

    ratedHubPower  = prob['machine_rating'] / maxEfficiency 
    rotorSpeed     = (maxTipSpd/(0.5*prob['rotor_diameter'])) * (60.0 / (2*np.pi))
    prob['rotor_thrust']  = airDensity * thrustCoeff * np.pi * prob['rotor_diameter']**2 * (ratedWindSpd**2) / 8
    prob['rotor_torque'] = ratedHubPower/(rotorSpeed*(np.pi/30))*1000
    
    prob['year'] = 2009
    prob['month'] = 12

    # AEP inputs
    prob['max_tip_speed'] = 80.0 #Float(units = 'm/s', iotype='in', desc= 'maximum allowable tip speed for the rotor')
    prob['max_power_coefficient'] = 0.488 #Float(iotype='in', desc= 'maximum power coefficient of rotor for operation in region 2')
    prob['opt_tsr'] = 7.525 #Float(iotype='in', desc= 'optimum tip speed ratio for operation in region 2')
    prob['cut_in_wind_speed'] = 3.0 #Float(units = 'm/s', iotype='in', desc= 'cut in wind speed for the wind turbine')
    prob['cut_out_wind_speed'] = 25.0 #Float(units = 'm/s', iotype='in', desc= 'cut out wind speed for the wind turbine')
    prob['altitude'] = 0.0 #Float(units = 'm', iotype='in', desc= 'altitude of wind plant')
    prob['air_density'] = 0 #Float(units = 'kg / (m * m * m)', iotype='in', desc= 'air density at wind plant site')  # default air density value is 0.0 - forces aero csm to calculate air density in model
    prob['max_efficiency'] = 0.902 #Float(iotype='in', desc = 'maximum efficiency of rotor and drivetrain - at rated power')
    prob['thrust_coefficient'] = 0.5 #Float(iotype='in', desc='thrust coefficient at rated power')
    prob['soiling_losses'] = 0.0
    prob['array_losses'] = 0.1
    prob['availability'] = 0.941
    prob['turbine_number'] = 100
    prob['shear_exponent'] = 0.1
    prob['wind_speed_50m'] = 8.02
    prob['weibull_k']= 2.15

    # Finance, BOS and OPEX inputs
    prob['RNA_mass'] = 256634.5 # RNA mass is not used in this simple model
    prob['sea_depth'] = 20.0
    prob['multiplier'] = 1.0

    prob.run()
    print("Overall cost of energy for an offshore wind plant with 100 NREL 5 MW turbines")
    for io in root.unknowns:
        print(io + ' ' + str(root.unknowns[io]))

if __name__=="__main__":

    example()
