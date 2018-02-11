"""
bos_csm_component.py

Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""
from fused_wind import FUSED_Object , FUSED_OpenMDAO , fusedvar
from windio_plant_costs import fifc_bos_costs

from openmdao.api import IndepVarComp, Component, Problem, Group

from config import *
import numpy as np

### FUSED-wrapper file (in WISDEM/Plant_CostsSE)
class bos_csm_fused(FUSED_Object):

    def __init__(self):

        super(bos_csm_fused, self).__init__()

        self.implement_fifc(fifc_bos_costs) # pulls in variables from fused-wind interface (not explicit)

        # Add model specific inputs
        self.add_input(**fusedvar('sea_depth',0.0)) # = Float(20.0, units = 'm', iotype = 'in', desc = 'sea depth for offshore wind plant')
        self.add_input(**fusedvar('year',0.0)) # = Int(2009, iotype='in', desc='year for project start')
        self.add_input(**fusedvar('month',0.0)) # = Int(12, iotype = 'in', desc= 'month for project start')
        self.add_input(**fusedvar('multiplier',0.0)) # = Float(1.0, iotype='in')

        # Add model specific outputs
        self.add_output(**fusedvar('bos_breakdown_development_costs',0.0)) #  = Float(desc='Overall wind plant balance of station/system costs up to point of comissioning')
        self.add_output(**fusedvar('bos_breakdown_preparation_and_staging_costs',0.0)) #  = Float(desc='Site preparation and staging')
        self.add_output(**fusedvar('bos_breakdown_transportation_costs',0.0)) #  = Float(desc='Any transportation costs to site / staging site') #BOS or turbine cost?
        self.add_output(**fusedvar('bos_breakdown_foundation_and_substructure_costs',0.0)) # Float(desc='Foundation and substructure costs')
        self.add_output(**fusedvar('bos_breakdown_electrical_costs',0.0)) # Float(desc='Collection system, substation, transmission and interconnect costs')
        self.add_output(**fusedvar('bos_breakdown_assembly_and_installation_costs',0.0)) # Float(desc='Assembly and installation costs')
        self.add_output(**fusedvar('bos_breakdown_soft_costs',0.0)) # = Float(desc='Contingencies, bonds, reserves, decommissioning, profits, and construction financing costs')
        self.add_output(**fusedvar('bos_breakdown_other_costs',0.0)) # = Float(desc='Bucket for any other costs not captured above')

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

        print(self.bos.bos_costs)
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

class bos_csm(object):

    def __init__(self):

        # Outputs
        #bos_breakdown = VarTree(BOSVarTree(), iotype='out', desc='BOS cost breakdown')
        #bos_costs = Float(iotype='out', desc='Overall wind plant balance of station/system costs up to point of comissioning')
        self.bos_costs = 0.0 # *= self.multiplier  # TODO: add to gradients
        self.bos_breakdown_development_costs = 0.0 # engPermits_costs * self.turbine_number
        self.bos_breakdown_preparation_and_staging_costs = 0.0 # (roadsCivil_costs + portStaging_costs) * self.turbine_number
        self.bos_breakdown_transportation_costs = 0.0 # (transportation_costs * self.turbine_number)
        self.bos_breakdown_foundation_and_substructure_costs = 0.0 # foundation_cost * self.turbine_number
        self.bos_breakdown_electrical_costs = 0.0 # electrical_costs * self.turbine_number
        self.bos_breakdown_assembly_and_installation_costs = 0.0 # installation_costs * self.turbine_number
        self.bos_breakdown_soft_costs = 0.0 # 0.0
        self.bos_breakdown_other_costs = 0.0 # (pai_costs + scour_costs + suretyBond) * self.turbine_number

    def compute(self, machine_rating, rotor_diameter, hub_height, RNA_mass, turbine_cost, turbine_number = 100, sea_depth = 20.0, year = 2009, month=12, multiplier = 1.0):

        # for coding ease
        # Default Variables
        self.machine_rating = machine_rating #Float(iotype='in', units='kW', desc='turbine machine rating')
        self.rotor_diameter= rotor_diameter #Float(iotype='in', units='m', desc='rotor diameter')
        self.hub_height = hub_height #Float(iotype='in', units='m', desc='hub height')
        self.RNA_mass = RNA_mass #Float(iotype='in', units='kg', desc='Rotor Nacelle Assembly mass')
        self.turbine_cost = turbine_cost #Float(iotype='in', units='USD', desc='Single Turbine Capital _costs')

        # Parameters
        self.turbine_number = turbine_number #Int(iotype='in', desc='number of turbines in project')
        self.sea_depth = sea_depth #Float(20.0, units = 'm', iotype = 'in', desc = 'sea depth for offshore wind plant')
        self.year = year #Int(2009, iotype='in', desc='year for project start')
        self.month = month #Int(12, iotype = 'in', desc= 'month for project start')
        self.multiplier = multiplier #Float(1.0, iotype='in')

        lPrmtsCostCoeff1 = 9.94E-04
        lPrmtsCostCoeff2 = 20.31
        oPrmtsCostFactor = 37.0 # $/kW (2003)
        scourCostFactor =  55.0 # $/kW (2003)
        ptstgCostFactor =  20.0 # $/kW (2003)
        ossElCostFactor = 260.0 # $/kW (2003) shallow
        ostElCostFactor = 290.0 # $/kW (2003) transitional
        ostSTransFactor  =  25.0 # $/kW (2003)
        ostTTransFactor  =  77.0 # $/kW (2003)
        osInstallFactor  = 100.0 # $/kW (2003) shallow & trans
        suppInstallFactor = 330.0 # $/kW (2003) trans additional
        paiCost         = 60000.0 # per turbine

        suretyBRate     = 0.03  # 3% of ICC
        suretyBond      = 0.0

        #set variables
        if self.sea_depth == 0:            # type of plant # 1: Land, 2: < 30m, 3: < 60m, 4: >= 60m
            iDepth = 1
        elif self.sea_depth < 30:
            iDepth = 2
        elif self.sea_depth < 60:
            iDepth = 3
        else:
            iDepth = 4

        # initialize self.ppi index calculator
        if iDepth == 1:
            ref_yr  = 2002
            ref_mon =    9
        else:
            ref_yr = 2003
            ref_mon = 9
        ppi.ref_yr = ref_yr
        ppi.ref_mon = ref_mon
        ppi.curr_yr = self.year
        ppi.curr_mon = self.month

        self.d_foundation_d_diameter = 0.0
        self.d_foundation_d_hheight = 0.0
        self.d_foundation_d_rating = 0.0
        # foundation costs
        if (iDepth == 1): # land
            fcCoeff = 303.23
            fcExp   = 0.4037
            SweptArea = (self.rotor_diameter*0.5)**2.0 * np.pi
            foundation_cost = fcCoeff * (self.hub_height*SweptArea)**fcExp
            fndnCostEscalator = ppi.compute('IPPI_FND')
            self.d_foundation_d_diameter = fndnCostEscalator * fcCoeff * fcExp * ((self.hub_height*(2.0 * 0.5 * (self.rotor_diameter * 0.5) * np.pi))**(fcExp-1)) * self.hub_height
            self.d_foundation_d_hheight = fndnCostEscalator * fcCoeff * fcExp * ((self.hub_height*SweptArea)**(fcExp-1)) * SweptArea
        elif (iDepth == 2):
            sscf = 300.0 # $/kW
            foundation_cost = sscf*self.machine_rating
            fndnCostEscalator = ppi.compute('IPPI_MPF')
            self.d_foundation_d_rating = fndnCostEscalator * sscf
        elif (iDepth == 3):
            sscf = 450.0 # $/kW
            foundation_cost = sscf*self.machine_rating
            fndnCostEscalator = ppi.compute('IPPI_OAI')
            self.d_foundation_d_rating = fndnCostEscalator * sscf
        elif (iDepth == 4):
            foundation_cost = 0.0
            fndnCostEscalator = 1.0

        foundation_cost *= fndnCostEscalator

        # cost calculations
        tpC1  =0.00001581
        tpC2  =-0.0375
        tpInt =54.7
        tFact = tpC1*self.machine_rating*self.machine_rating + tpC2*self.machine_rating + tpInt

        roadsCivil_costs = 0.0
        portStaging_costs = 0.0
        pai_costs = 0.0
        scour_costs = 0.0
        self.d_assembly_d_diameter = 0.0
        self.d_assembly_d_hheight = 0.0
        self.d_development_d_rating = 0.0
        self.d_preparation_d_rating = 0.0
        self.d_transport_d_rating = 0.0
        self.d_electrical_d_rating = 0.0
        self.d_assembly_d_rating = 0.0
        self.d_other_d_rating = 0.0
        if (iDepth == 1):
            engPermits_costs  = (lPrmtsCostCoeff1 * self.machine_rating * self.machine_rating) + \
                               (lPrmtsCostCoeff2 * self.machine_rating)
            ppi.ref_mon = 3
            engPermits_costs *= ppi.compute('IPPI_LPM')
            self.d_development_d_rating = ppi.compute('IPPI_LPM') * (2.0 * lPrmtsCostCoeff1 * self.machine_rating + lPrmtsCostCoeff2)
            ppi.ref_mon = 9

            elC1  = 3.49E-06
            elC2  = -0.0221
            elInt = 109.7
            eFact = elC1*self.machine_rating*self.machine_rating + elC2*self.machine_rating + elInt
            electrical_costs = self.machine_rating * eFact * ppi.compute('IPPI_LEL')
            self.d_electrical_d_rating = ppi.compute('IPPI_LEL') * (3. * elC1*self.machine_rating**2. + \
                                    2. * elC2*self.machine_rating + elInt)

            rcC1  = 2.17E-06
            rcC2  = -0.0145
            rcInt =69.54
            rFact = rcC1*self.machine_rating*self.machine_rating + rcC2*self.machine_rating + rcInt
            roadsCivil_costs = self.machine_rating * rFact * ppi.compute('IPPI_RDC')
            self.d_preparation_d_rating = ppi.compute('IPPI_RDC') * (3. * rcC1 * self.machine_rating**2. + \
                                     2. * rcC2 * self.machine_rating + rcInt)

            iCoeff = 1.965
            iExp   = 1.1736
            installation_costs = iCoeff * ((self.hub_height*self.rotor_diameter)**iExp) * ppi.compute('IPPI_LAI')
            self.d_assembly_d_diameter = iCoeff * ((self.hub_height*self.rotor_diameter)**(iExp-1)) * self.hub_height * ppi.compute('IPPI_LAI')
            self.d_assembly_d_hheight = iCoeff * ((self.hub_height*self.rotor_diameter)**(iExp-1)) * self.rotor_diameter * ppi.compute('IPPI_LAI')

            transportation_costs = self.machine_rating * tFact * ppi.compute('IPPI_TPT')
            self.d_transport_d_rating = ppi.compute('IPPI_TPT') * (tpC1* 3. * self.machine_rating**2. + \
                                   tpC2* 2. * self.machine_rating + tpInt )

        elif (iDepth == 2):  # offshore shallow
            ppi.ref_yr = 2003
            pai_costs            = paiCost * ppi.compute('IPPI_PAE')
            portStaging_costs    = ptstgCostFactor  * self.machine_rating * ppi.compute('IPPI_STP') # 1.415538133
            self.d_preparation_d_rating = ptstgCostFactor * ppi.compute('IPPI_STP')
            engPermits_costs     = oPrmtsCostFactor * self.machine_rating * ppi.compute('IPPI_OPM')
            self.d_development_d_rating = oPrmtsCostFactor * ppi.compute('IPPI_OPM')
            scour_costs         = scourCostFactor  * self.machine_rating * ppi.compute('IPPI_STP') # 1.415538133#
            self.d_other_d_rating = scourCostFactor  * ppi.compute('IPPI_STP')
            installation_costs   = osInstallFactor  * self.machine_rating * ppi.compute('IPPI_OAI')
            self.d_assembly_d_rating = osInstallFactor * ppi.compute('IPPI_OAI')
            electrical_costs     = ossElCostFactor  * self.machine_rating * ppi.compute('IPPI_OEL')
            self.d_electrical_d_rating = ossElCostFactor  * ppi.compute('IPPI_OEL')
            ppi.ref_yr  = 2002
            transportation_costs = self.machine_rating * tFact * ppi.compute('IPPI_TPT')
            self.d_transport_d_rating = ppi.compute('IPPI_TPT') * (tpC1* 3. * self.machine_rating**2. + \
                                   tpC2* 2. * self.machine_rating + tpInt )
            ppi.ref_yr = 2003

        elif (iDepth == 3):  # offshore transitional depth
            ppi.ref_yr = 2003
            turbInstall   = osInstallFactor  * self.machine_rating * ppi.compute('IPPI_OAI')
            supportInstall = suppInstallFactor * self.machine_rating * ppi.compute('IPPI_OAI')
            installation_costs = turbInstall + supportInstall
            self.d_assembly_d_rating = (osInstallFactor + suppInstallFactor) * ppi.compute('IPPI_OAI')
            pai_costs          = paiCost                          * ppi.compute('IPPI_PAE')
            electrical_costs     = ostElCostFactor  * self.machine_rating * ppi.compute('IPPI_OEL')
            self.d_electrical_d_rating = ossElCostFactor  * ppi.compute('IPPI_OEL')
            portStaging_costs   = ptstgCostFactor  * self.machine_rating * ppi.compute('IPPI_STP')
            self.d_preparation_d_rating = ptstgCostFactor * ppi.compute('IPPI_STP')
            engPermits_costs     = oPrmtsCostFactor * self.machine_rating * ppi.compute('IPPI_OPM')
            self.d_development_d_rating = oPrmtsCostFactor * ppi.compute('IPPI_OPM')
            scour_costs          = scourCostFactor  * self.machine_rating * ppi.compute('IPPI_STP')
            self.d_other_d_rating = scourCostFactor * ppi.compute('IPPI_STP')
            ppi.ref_yr  = 2002
            turbTrans           = ostTTransFactor  * self.machine_rating * ppi.compute('IPPI_TPT')
            self.d_transport_d_rating = ostTTransFactor  * ppi.compute('IPPI_TPT')
            ppi.ref_yr = 2003
            supportTrans        = ostSTransFactor  * self.machine_rating * ppi.compute('IPPI_OAI')
            transportation_costs = turbTrans + supportTrans
            self.d_transport_d_rating += ostSTransFactor  * ppi.compute('IPPI_OAI')

        elif (iDepth == 4):  # offshore deep
            print("\ncsmBOS: Add costCat 4 code\n\n")

        bos_costs = foundation_cost + \
                    transportation_costs + \
                    roadsCivil_costs    + \
                    portStaging_costs   + \
                    installation_costs   + \
                    electrical_costs     + \
                    engPermits_costs    + \
                    pai_costs          + \
                    scour_costs

        self.d_other_d_tcc = 0.0
        if (self.sea_depth > 0.0):
            suretyBond = suretyBRate * (self.turbine_cost + bos_costs)
            self.d_other_d_tcc = suretyBRate
            d_surety_d_rating = suretyBRate * (self.d_development_d_rating + self.d_preparation_d_rating + self.d_transport_d_rating + \
                          self.d_foundation_d_rating + self.d_electrical_d_rating + self.d_assembly_d_rating + self.d_other_d_rating)
            self.d_other_d_rating += d_surety_d_rating
        else:
            suretyBond = 0.0

        self.bos_costs = self.turbine_number * (bos_costs + suretyBond)
        self.bos_costs *= self.multiplier  # TODO: add to gradients

        self.bos_breakdown_development_costs = engPermits_costs * self.turbine_number
        self.bos_breakdown_preparation_and_staging_costs = (roadsCivil_costs + portStaging_costs) * self.turbine_number
        self.bos_breakdown_transportation_costs = (transportation_costs * self.turbine_number)
        self.bos_breakdown_foundation_and_substructure_costs = foundation_cost * self.turbine_number
        self.bos_breakdown_electrical_costs = electrical_costs * self.turbine_number
        self.bos_breakdown_assembly_and_installation_costs = installation_costs * self.turbine_number
        self.bos_breakdown_soft_costs = 0.0
        self.bos_breakdown_other_costs = (pai_costs + scour_costs + suretyBond) * self.turbine_number

        # derivatives
        self.d_development_d_rating *= self.turbine_number
        self.d_preparation_d_rating *= self.turbine_number
        self.d_transport_d_rating *= self.turbine_number
        self.d_foundation_d_rating *= self.turbine_number
        self.d_electrical_d_rating *= self.turbine_number
        self.d_assembly_d_rating *= self.turbine_number
        self.d_soft_d_rating = 0.0
        self.d_other_d_rating *= self.turbine_number
        self.d_cost_d_rating = self.d_development_d_rating + self.d_preparation_d_rating + self.d_transport_d_rating + \
                          self.d_foundation_d_rating + self.d_electrical_d_rating + self.d_assembly_d_rating + \
                          self.d_soft_d_rating + self.d_other_d_rating

        self.d_development_d_diameter = 0.0
        self.d_preparation_d_diameter = 0.0
        self.d_transport_d_diameter = 0.0
        #self.d_foundation_d_diameter
        self.d_electrical_d_diameter = 0.0
        #self.d_assembly_d_diameter
        self.d_soft_d_diameter = 0.0
        self.d_other_d_diameter = 0.0
        self.d_cost_d_diameter = self.d_development_d_diameter + self.d_preparation_d_diameter + self.d_transport_d_diameter + \
                          self.d_foundation_d_diameter + self.d_electrical_d_diameter + self.d_assembly_d_diameter + \
                          self.d_soft_d_diameter + self.d_other_d_diameter

        self.d_development_d_tcc = 0.0
        self.d_preparation_d_tcc = 0.0
        self.d_transport_d_tcc = 0.0
        self.d_foundation_d_tcc = 0.0
        self.d_electrical_d_tcc = 0.0
        self.d_assembly_d_tcc = 0.0
        self.d_soft_d_tcc = 0.0
        self.d_other_d_tcc *= self.turbine_number
        self.d_cost_d_tcc = self.d_development_d_tcc + self.d_preparation_d_tcc + self.d_transport_d_tcc + \
                          self.d_foundation_d_tcc + self.d_electrical_d_tcc + self.d_assembly_d_tcc + \
                          self.d_soft_d_tcc + self.d_other_d_tcc

        self.d_development_d_hheight = 0.0
        self.d_preparation_d_hheight = 0.0
        self.d_transport_d_hheight = 0.0
        #self.d_foundation_d_hheight
        self.d_electrical_d_hheight = 0.0
        #self.d_assembly_d_hheight
        self.d_soft_d_hheight = 0.0
        self.d_other_d_hheight = 0.0
        self.d_cost_d_hheight = self.d_development_d_hheight + self.d_preparation_d_hheight + self.d_transport_d_hheight + \
                          self.d_foundation_d_hheight + self.d_electrical_d_hheight + self.d_assembly_d_hheight + \
                          self.d_soft_d_hheight + self.d_other_d_hheight

        self.d_development_d_rna = 0.0
        self.d_preparation_d_rna = 0.0
        self.d_transport_d_rna = 0.0
        self.d_foundation_d_rna = 0.0
        self.d_electrical_d_rna = 0.0
        self.d_assembly_d_rna = 0.0
        self.d_soft_d_rna = 0.0
        self.d_other_d_rna = 0.0
        self.d_cost_d_rna = self.d_development_d_rna + self.d_preparation_d_rna + self.d_transport_d_rna + \
                          self.d_foundation_d_rna + self.d_electrical_d_rna + self.d_assembly_d_rna + \
                          self.d_soft_d_rna + self.d_other_d_rna

    def list_deriv_vars(self):

        inputs = ['machine_rating', 'rotor_diameter', 'turbine_cost', 'hub_height', 'RNA_mass']
        outputs = ['bos_breakdown.development_costs', 'bos_breakdown.preparation_and_staging_costs',\
                   'bos_breakdown.transportation_costs', 'bos_breakdown.foundation_and_substructure_costs',\
                   'bos_breakdown.electrical_costs', 'bos_breakdown.assembly_and_installation_costs',\
                   'bos_breakdown.soft_costs', 'bos_breakdown.other_costs', 'bos_costs']

        return inputs, outputs

    def provideJ(self):

        self.J = np.array([[self.d_development_d_rating, self.d_development_d_diameter, self.d_development_d_tcc, self.d_development_d_hheight, self.d_development_d_rna],\
                           [self.d_preparation_d_rating, self.d_preparation_d_diameter, self.d_preparation_d_tcc, self.d_preparation_d_hheight, self.d_preparation_d_rna],\
                           [self.d_transport_d_rating, self.d_transport_d_diameter, self.d_transport_d_tcc, self.d_transport_d_hheight, self.d_transport_d_rna],\
                           [self.d_foundation_d_rating, self.d_foundation_d_diameter, self.d_foundation_d_tcc, self.d_foundation_d_hheight, self.d_foundation_d_rna],\
                           [self.d_electrical_d_rating, self.d_electrical_d_diameter, self.d_electrical_d_tcc, self.d_electrical_d_hheight, self.d_electrical_d_rna],\
                           [self.d_assembly_d_rating, self.d_assembly_d_diameter, self.d_assembly_d_tcc, self.d_assembly_d_hheight, self.d_assembly_d_rna],\
                           [self.d_soft_d_rating, self.d_soft_d_diameter, self.d_soft_d_tcc, self.d_soft_d_hheight, self.d_soft_d_rna],\
                           [self.d_other_d_rating, self.d_other_d_diameter, self.d_other_d_tcc, self.d_other_d_hheight, self.d_other_d_rna],\
                           [self.d_cost_d_rating, self.d_cost_d_diameter, self.d_cost_d_tcc, self.d_cost_d_hheight, self.d_cost_d_rna]])

        return self.J

def example():

    # openmdao example of execution
    root = Group()
    root.add('bos_csm_test', FUSED_OpenMDAO(bos_csm_fused()), promotes=['*'])
    prob = Problem(root)
    prob.setup()

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

    prob.run()
    print("Balance of Station Costs for an offshore wind plant with 100 NREL 5 MW turbines")
    for io in root.unknowns:
        print(io + ' ' + str(root.unknowns[io]))

    prob['sea_depth'] = 0.0
    prob['turbine_cost'] = 5229222.77

    prob.run()
    print("Balance of Station Costs for an land-based wind plant with 100 NREL 5 MW turbines")
    for io in root.unknowns:
        print(io + ' ' + str(root.unknowns[io]))

if __name__ == "__main__":

    example()
