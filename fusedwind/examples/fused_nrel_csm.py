"""
aero_csm_component.py

Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""

# NREL CSM model set
from nrelcsm.nrel_csm import aep_csm, tcc_csm, bos_csm, opex_csm, fin_csm

# FUSED helper functions and interface defintions
from fusedwind.fused_wind import create_interface , FUSED_Object , set_output, set_input #, fusedvar
from fusedwind.windio_plant_costs import fifc_aep, fifc_tcc_costs, fifc_bos_costs, fifc_opex, fifc_finance

import numpy as np

### FUSED-wrapper file 
class aep_csm_fused(FUSED_Object):

    def __init__(self):
        super(aep_csm_fused, self).__init__()

        self.implement_fifc(fifc_aep) # pulls in variables from fused-wind interface (not explicit)

        # Add model specific inputs
        self.add_input(**{'name': 'max_tip_speed', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'max_power_coefficient', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'opt_tsr', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'cut_in_wind_speed', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'cut_out_wind_speed', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'air_density', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'max_efficiency', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'thrust_coefficient', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'soiling_losses', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'array_losses', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'availability', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'shear_exponent', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'wind_speed_50m', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'weibull_k', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'hub_height', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'altitude', 'val' : 0.0, 'type' : float})

        # Add model specific outputs
        self.add_output(**{'name': 'rated_wind_speed', 'val' : 0.0, 'type' : float})
        self.add_output(**{'name': 'rated_rotor_speed', 'val' : 0.0, 'type' : float})
        self.add_output(**{'name': 'rotor_thrust', 'val' : 0.0, 'type' : float})
        self.add_output(**{'name': 'rotor_torque', 'val' : 0.0, 'type' : float})
        self.add_output(**{'name': 'power_curve', 'val' : np.zeros(161), 'type' : float, 'shape' : (161,)})
        self.add_output(**{'name': 'gross_aep', 'val' : 0.0, 'type' : float})
        self.add_output(**{'name': 'capacity_factor', 'val' : 0.0, 'type' : float})

        self.aep_csm_assembly = aep_csm()

    def compute(self, inputs, outputs):
        self.aep_csm_assembly.compute(inputs['machine_rating'], inputs['max_tip_speed'], inputs['rotor_diameter'], inputs['max_power_coefficient'], inputs['opt_tsr'],
                inputs['cut_in_wind_speed'], inputs['cut_out_wind_speed'], inputs['hub_height'], inputs['altitude'], inputs['air_density'],
                inputs['max_efficiency'], inputs['thrust_coefficient'], inputs['soiling_losses'], inputs['array_losses'], inputs['availability'],
                inputs['turbine_number'], inputs['shear_exponent'], inputs['wind_speed_50m'], inputs['weibull_k'])

        outputs['rated_wind_speed'] = self.aep_csm_assembly.aero.rated_wind_speed
        outputs['rated_rotor_speed'] = self.aep_csm_assembly.aero.rated_rotor_speed
        outputs['rotor_thrust'] = self.aep_csm_assembly.aero.rotor_thrust
        outputs['rotor_torque'] = self.aep_csm_assembly.aero.rotor_torque
        outputs['power_curve'] = self.aep_csm_assembly.aero.power_curve
        outputs['gross_aep'] = self.aep_csm_assembly.aep.gross_aep
        outputs['net_aep'] = self.aep_csm_assembly.aep.net_aep
        outputs['capacity_factor'] = self.aep_csm_assembly.aep.capacity_factor


### FUSED-wrapper file 
class tcc_csm_fused(FUSED_Object):

    def __init__(self, offshore=False, advanced_blade=True, drivetrain_design='geared', \
                       crane=True, advanced_bedplate=0, advanced_tower=False):

        super(tcc_csm_fused, self).__init__()

        self.offshore = offshore 
        self.advanced_blade = advanced_blade 
        self.drivetrain_design = drivetrain_design 
        self.crane = crane 
        self.advanced_bedplate = advanced_bedplate  
        self.advanced_tower = advanced_tower

        self.implement_fifc(fifc_tcc_costs) # pulls in variables from fused-wind interface (not explicit)

        # Add model specific inputs
        self.add_input(**{'name': 'rotor_thrust', 'val' : 0.0, 'type' : float})
        self.add_input(**{'name': 'rotor_torque', 'val' : 0.0, 'type' : float}) 
        self.add_input(**{'name': 'year', 'val' : 2010, 'type' : int})
        self.add_input(**{'name': 'month', 'val' : 12, 'type' : int})

        # Add model specific outputs
        self.add_output(**{'name': 'rotor_cost', 'val' : 0.0, 'type' : float})
        self.add_output(**{'name': 'rotor_mass', 'val' : 0.0, 'type' : float}) 
        self.add_output(**{'name': 'turbine_mass', 'val' : 0.0, 'type' : float}) 

        self.tcc = tcc_csm()

    def compute(self, inputs, outputs):

        machine_rating = inputs['machine_rating']
        rotor_diameter = inputs['rotor_diameter']
        hub_height = inputs['hub_height']
        blade_number = inputs['blade_number']
        rotor_thrust = inputs['rotor_thrust']
        rotor_torque = inputs['rotor_torque']
        year = inputs['year']
        month = inputs['month']

        self.tcc.compute(rotor_diameter, machine_rating, hub_height, rotor_thrust, rotor_torque, \
                year, month, blade_number, self.offshore, self.advanced_blade, self.drivetrain_design, \
                self.crane, self.advanced_bedplate, self.advanced_tower)

        # Outputs
        outputs['turbine_cost'] = self.tcc.turbine_cost 
        outputs['turbine_mass'] = self.tcc.turbine_mass
        outputs['rotor_cost'] = self.tcc.rotor_cost
        outputs['rotor_mass'] = self.tcc.rotor_mass
        
        return outputs


### FUSED-wrapper file 
class bos_csm_fused(FUSED_Object):

    def __init__(self):

        super(bos_csm_fused, self).__init__()

        self.implement_fifc(fifc_bos_costs) # pulls in variables from fused-wind interface (not explicit)

        # Add model specific inputs
        self.add_input(**{'name': 'sea_depth', 'val' : 0.0, 'type' : float}) # = Float(20.0, units = 'm', iotype = 'in', desc = 'sea depth for offshore wind plant')
        self.add_input(**{'name': 'year', 'val' : 2010, 'type' : int}) # = Int(2009, iotype='in', desc='year for project start')
        self.add_input(**{'name': 'month', 'val' : 12, 'type' : int}) # = Int(12, iotype = 'in', desc= 'month for project start')
        self.add_input(**{'name': 'multiplier', 'val' : 0.0, 'type' : float}) # = Float(1.0, iotype='in')

        # Add model specific outputs
        self.add_output(**{'name': 'bos_breakdown_development_costs', 'val' : 0.0, 'type' : float}) #  = Float(desc='Overall wind plant balance of station/system costs up to point of comissioning')
        self.add_output(**{'name': 'bos_breakdown_preparation_and_staging_costs', 'val' : 0.0, 'type' : float}) #  = Float(desc='Site preparation and staging')
        self.add_output(**{'name': 'bos_breakdown_transportation_costs', 'val' : 0.0, 'type' : float}) #  = Float(desc='Any transportation costs to site / staging site') #BOS or turbine cost?
        self.add_output(**{'name': 'bos_breakdown_foundation_and_substructure_costs', 'val' : 0.0, 'type' : float}) # Float(desc='Foundation and substructure costs')
        self.add_output(**{'name': 'bos_breakdown_electrical_costs', 'val' : 0.0, 'type' : float}) # Float(desc='Collection system, substation, transmission and interconnect costs')
        self.add_output(**{'name': 'bos_breakdown_assembly_and_installation_costs', 'val' : 0.0, 'type' : float}) # Float(desc='Assembly and installation costs')
        self.add_output(**{'name': 'bos_breakdown_soft_costs', 'val' : 0.0, 'type' : float}) # = Float(desc='Contingencies, bonds, reserves, decommissioning, profits, and construction financing costs')
        self.add_output(**{'name': 'bos_breakdown_other_costs', 'val' : 0.0, 'type' : float}) # = Float(desc='Bucket for any other costs not captured above')

        self.bos = bos_csm()

    def compute(self, inputs, outputs):

        machine_rating = inputs['machine_rating']
        rotor_diameter = inputs['rotor_diameter']
        hub_height = inputs['hub_height']
        RNA_mass = inputs['RNA_mass']
        turbine_cost = inputs['turbine_cost']

        turbine_number = inputs['turbine_number']
        sea_depth = inputs['sea_depth']
        year = inputs['year']
        month = inputs['month']
        multiplier = inputs['multiplier']

        self.bos.compute(machine_rating, rotor_diameter, hub_height, RNA_mass, turbine_cost, turbine_number, sea_depth, year, month, multiplier)

        # Outputs
        outputs['bos_costs'] = self.bos.bos_costs #  = Float(iotype='out', desc='Overall wind plant balance of station/system costs up to point of comissioning')
        #self.add_output(bos_breakdown = VarTree(BOSVarTree(), iotype='out', desc='BOS cost breakdown')
        outputs['bos_breakdown_development_costs'] = self.bos.bos_breakdown_development_costs #  = Float(desc='Overall wind plant balance of station/system costs up to point of comissioning')
        outputs['bos_breakdown_preparation_and_staging_costs'] = self.bos.bos_breakdown_preparation_and_staging_costs #  = Float(desc='Site preparation and staging')
        outputs['bos_breakdown_transportation_costs'] = self.bos.bos_breakdown_transportation_costs #  = Float(desc='Any transportation costs to site / staging site') #BOS or turbine cost?
        outputs['bos_breakdown_foundation_and_substructure_costs'] = self.bos.bos_breakdown_foundation_and_substructure_costs # Float(desc='Foundation and substructure costs')
        outputs['bos_breakdown_electrical_costs'] = self.bos.bos_breakdown_electrical_costs # Float(desc='Collection system, substation, transmission and interconnect costs')
        outputs['bos_breakdown_assembly_and_installation_costs'] = self.bos.bos_breakdown_assembly_and_installation_costs # Float(desc='Assembly and installation costs')
        outputs['bos_breakdown_soft_costs'] = self.bos.bos_breakdown_soft_costs  # = Float(desc='Contingencies, bonds, reserves, decommissioning, profits, and construction financing costs')
        outputs['bos_breakdown_other_costs'] = self.bos.bos_breakdown_other_costs # = Float(desc='Bucket for any other costs not captured above')

        return outputs


### FUSED-wrapper file 
class opex_csm_fused(FUSED_Object):

    def __init__(self):
        super(opex_csm_fused, self).__init__()

        self.implement_fifc(fifc_opex)

        # Add model specific inputs
        self.add_input(**{'name': 'sea_depth', 'val' : 0.0, 'type' : float}) # #20.0, units = 'm', iotype = 'in', desc = 'sea depth for offshore wind plant')
        self.add_input(**{'name': 'year', 'val' : 2010, 'type' : int}) # = Int(2009, iotype='in', desc='year for project start')
        self.add_input(**{'name': 'month', 'val' : 12, 'type' : int}) # iotype = 'in', desc= 'month for project start') # units = months
        self.add_input(**{'name': 'net_aep', 'val' : 0.0, 'type' : float}) # units = 'kW * h', iotype = 'in', desc = 'annual energy production for the plant')

        # Add model specific outputs
        self.add_output(**{'name': 'opex_breakdown_preventative_opex', 'val' : 0.0, 'type' : float}) # desc='annual expenditures on preventative maintenance - BOP and turbines'
        self.add_output(**{'name': 'opex_breakdown_corrective_opex', 'val' : 0.0, 'type' : float}) # desc='annual unscheduled maintenance costs (replacements) - BOP and turbines'
        self.add_output(**{'name': 'opex_breakdown_lease_opex', 'val' : 0.0, 'type' : float}) # desc='annual lease expenditures'
        self.add_output(**{'name': 'opex_breakdown_other_opex', 'val' : 0.0, 'type' : float}) # desc='other operational expenditures such as fixed costs'

        self.opex = opex_csm()

    def compute(self, inputs, outputs):

        self.opex.compute(inputs['sea_depth'], inputs['year'], inputs['month'],
                          inputs['turbine_number'], inputs['machine_rating'], inputs['net_aep'])

        outputs['avg_annual_opex'] = self.opex.avg_annual_opex
        outputs['opex_breakdown_preventative_opex'] = self.opex.opex_breakdown_preventative_opex
        outputs['opex_breakdown_corrective_opex'] = self.opex.opex_breakdown_corrective_opex
        outputs['opex_breakdown_lease_opex'] = self.opex.opex_breakdown_lease_opex
        outputs['opex_breakdown_other_opex'] = self.opex.opex_breakdown_other_opex


### FUSED-wrapper file 
class fin_csm_fused(FUSED_Object):

    def __init__(self,fixed_charge_rate = 0.12, construction_finance_rate=0.0, tax_rate = 0.4, discount_rate = 0.07, \
                      construction_time = 1.0, project_lifetime = 20.0, sea_depth = 20.0):

        super(fin_csm_fused, self).__init__()

        self.implement_fifc(fifc_finance) # pulls in variables from fused-wind interface (not explicit)

        self.add_input(**{'name': 'sea_depth', 'val' : 0.0, 'type' : float}) # #20.0, units = 'm', iotype = 'in', desc = 'sea depth for offshore wind plant')

        self.add_output(**{'name': 'lcoe', 'val' : 0.0, 'type' : float}) 
        
        self.fin = fin_csm(fixed_charge_rate, construction_finance_rate, tax_rate, discount_rate, \
                      construction_time, project_lifetime)

    def compute(self, inputs, outputs):

        turbine_cost = inputs['turbine_cost']
        turbine_number = inputs['turbine_number']
        bos_costs = inputs['bos_costs']
        avg_annual_opex = inputs['avg_annual_opex']
        net_aep = inputs['net_aep']
        sea_depth = inputs['sea_depth']

        self.fin.compute(turbine_cost, turbine_number, bos_costs, avg_annual_opex, net_aep, sea_depth)

        # Outputs
        outputs['coe'] = self.fin.coe 
        outputs['lcoe'] = self.fin.lcoe 

        return outputs