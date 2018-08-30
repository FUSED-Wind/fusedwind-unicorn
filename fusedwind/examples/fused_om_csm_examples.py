"""
FUSED LCOE assmbly example.
"""

# FUSED wrapper
from fusedwind.fused_openmdao import FUSED_Component, FUSED_Group, FUSED_add, FUSED_print, \
                                     FUSED_Problem, FUSED_setup, FUSED_run, FUSED_VarComp, FUSED_OpenMDAOBase

# NREL cost and scaling model sub-assemblies
from fusedwind.examples.fused_nrel_csm import tcc_csm_fused, bos_csm_fused, opex_csm_fused, fin_csm_fused, aep_csm_fused

import numpy as np

### examples for individual fused components
############################################

def example_aep():

    root = FUSED_Group()
    FUSED_add(root, 'aep_test', FUSED_Component(aep_csm_fused()), ['*'])

    prob = FUSED_Problem(root)
    FUSED_OpenMDAOBase.setup_splits()
    FUSED_setup(prob)

    prob['machine_rating'] = 5000.0 #Float(units = 'kW', iotype='in', desc= 'rated machine power in kW')
    prob['max_tip_speed'] = 80.0 #Float(units = 'm/s', iotype='in', desc= 'maximum allowable tip speed for the rotor')
    prob['rotor_diameter'] = 126.0 #Float(units = 'm', iotype='in', desc= 'rotor diameter of the machine')
    prob['max_power_coefficient'] = 0.488 #Float(iotype='in', desc= 'maximum power coefficient of rotor for operation in region 2')
    prob['opt_tsr'] = 7.525 #Float(iotype='in', desc= 'optimum tip speed ratio for operation in region 2')
    prob['cut_in_wind_speed'] = 3.0 #Float(units = 'm/s', iotype='in', desc= 'cut in wind speed for the wind turbine')
    prob['cut_out_wind_speed'] = 25.0 #Float(units = 'm/s', iotype='in', desc= 'cut out wind speed for the wind turbine')
    prob['hub_height'] = 90.0 #Float(units = 'm', iotype='in', desc= 'hub height of wind turbine above ground / sea level')
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

    FUSED_run(prob)

    print("AEP output")
    FUSED_print(root)

def example_turbine():

    # openmdao example of execution
    root = FUSED_Group()
    FUSED_add(root, 'tcc_csm_test', FUSED_Component(tcc_csm_fused()), ['*'])
    prob = FUSED_Problem(root)
    FUSED_OpenMDAOBase.setup_splits()
    FUSED_setup(prob)

    # simple test of module
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

    FUSED_run(prob)
    
    print("The results for the NREL 5 MW Reference Turbine in an offshore 20 m water depth location are:")
    FUSED_print(root)

def example_bos():

    # openmdao example of execution
    root = FUSED_Group()
    FUSED_add(root, 'bos_csm_test', FUSED_Component(bos_csm_fused()), ['*'])
    prob = FUSED_Problem(root)
    FUSED_OpenMDAOBase.setup_splits()
    FUSED_setup(prob)

    prob['machine_rating'] = 5000.0
    prob['rotor_diameter'] = 126.0
    prob['turbine_cost'] = 5950209.28
    prob['hub_height'] = 90.0
    prob['RNA_mass'] = 256634.5 # RNA mass is not used in this simple model
    prob['turbine_number'] = 100
    prob['sea_depth'] = 20.0
    prob['year'] = 2009
    prob['month'] = 12
    prob['multiplier'] = 1.0

    FUSED_run(prob)
    print("Balance of Station Costs for an offshore wind plant with 100 NREL 5 MW turbines")
    FUSED_print(root)

    prob['sea_depth'] = 0.0
    prob['turbine_cost'] = 5229222.77

    FUSED_run(prob)
    print("Balance of Station Costs for an land-based wind plant with 100 NREL 5 MW turbines")
    FUSED_print(root)

def example_opex():

    # simple test of module

    root = FUSED_Group()
    FUSED_add(root, 'bos_opex_test', FUSED_Component(opex_csm_fused()), ['*'])

    prob = FUSED_Problem(root)
    FUSED_OpenMDAOBase.setup_splits()
    FUSED_setup(prob)

    prob['machine_rating'] = 5000.0 # Need to manipulate input or underlying cprob[onent will not execute
    prob['net_aep'] = 1701626526.28
    prob['sea_depth'] = 20.0
    prob['year'] = 2009
    prob['month'] = 12
    prob['turbine_number'] = 100

    FUSED_run(prob)
    print("Average annual operational expenditures for an offshore wind plant with 100 NREL 5 MW turbines")
    FUSED_print(root)

    prob['sea_depth'] = 0.0
    FUSED_run(prob)
    print("Average annual operational expenditures for an land-based wind plant with 100 NREL 5 MW turbines")
    FUSED_print(root)

def example_finance():

    # simple test of module

    # openmdao example of execution
    root = FUSED_Group()
    FUSED_add(root, 'fin_csm_test', FUSED_Component(fin_csm_fused(fixed_charge_rate = 0.12, construction_finance_rate=0.0, tax_rate = 0.4, discount_rate = 0.07, \
                      construction_time = 1.0, project_lifetime = 20.0, sea_depth = 20.0)), ['*'])
    prob = FUSED_Problem(root)
    FUSED_OpenMDAOBase.setup_splits()
    FUSED_setup(prob)

    prob['turbine_cost'] = 6087803.555 / 50
    prob['turbine_number'] = 50
    preventative_opex = 401819.023
    lease_opex = 22225.395
    corrective_opex = 91048.387
    prob['avg_annual_opex'] = preventative_opex + corrective_opex + lease_opex
    prob['bos_costs'] = 7668775.3
    prob['net_aep'] = 15756299.843
    prob['sea_depth'] = 20.0

    FUSED_run(prob)
    print("Overall cost of energy for an offshore wind plant with 100 NREL 5 MW turbines")
    FUSED_print(root)


### Full NREL cost and scaling model LCOE assembly and problem execution
#########################################################################

def example_lcoe():

    # openmdao example of execution
    root = FUSED_Group()
    FUSED_add(root, 'desvars',FUSED_VarComp([('machine_rating',5000.0),
                                                                     ('rotor_diameter', 126.0),
                                                                     ('hub_height', 90.0),
                                                                     ('turbine_number', 100.0),
                                                                     ('year', 2009.0),
                                                                     ('month',12.0),
                                                                     ('sea_depth', 20.0),
                                                                     ]), ['*'])

    FUSED_add(root, 'tcc_csm_test', FUSED_Component(tcc_csm_fused()), ['*'])
    FUSED_add(root, 'aep_test', FUSED_Component(aep_csm_fused()), ['*'])
    FUSED_add(root, 'bos_csm_test', FUSED_Component(bos_csm_fused()), ['*'])
    FUSED_add(root, 'opex_csm_test', FUSED_Component(opex_csm_fused()), ['*'])
    FUSED_add(root, 'fin_csm_test', FUSED_Component(fin_csm_fused()), ['*'])

    prob = FUSED_Problem(root)

    FUSED_OpenMDAOBase.setup_splits()
    FUSED_setup(prob)

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
    prob['air_density'] = 1.225 #Float(units = 'kg / (m * m * m)', iotype='in', desc= 'air density at wind plant site')  # default air density value is 0.0 - forces aero csm to calculate air density in model
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

    FUSED_run(prob)
    print("Overall cost of energy for an offshore wind plant with 100 NREL 5 MW turbines")
    FUSED_print(root)


if __name__=="__main__":

    example_aep()

    example_turbine()

    example_bos()

    example_opex()

    example_finance()

    example_lcoe()

    # print('--------------------- aep --------------------------------')
    # try:
    #     example_aep()
    # except Exception as e:
    #     print('FAILED')
    #     print(e)
    # 
    # print('--------------------- turbine --------------------------------')
    # example_turbine()
    # try:
    #     example_turbine()
    # except Exception as e:
    #     print('FAILED')
    #     print(e)
    # 
    # print('--------------------- bos --------------------------------')
    # try:
    #     example_bos()
    # except Exception as e:
    #     print('FAILED')
    #     print(e)

    # print('--------------------- opex --------------------------------')
    # try:
    #     example_opex()
    # except Exception as e:
    #     print('FAILED')
    #     print(e)
    # 
    # print('--------------------- finance --------------------------------')
    # try:
    #     example_finance()
    # except Exception as e:
    #     print('FAILED')
    #     print(e)
    # 
    # print('--------------------- lcoe --------------------------------')
    # try:
    #     example_lcoe()
    # except Exception as e:
    #     print('FAILED')
    #     print(e)

