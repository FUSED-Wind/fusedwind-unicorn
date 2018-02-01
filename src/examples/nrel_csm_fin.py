"""
fin_csm_component.py

Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""

from fused_wind import FUSED_Object , FUSED_OpenMDAO , fusedvar
from windio_plant_costs import fifc_finance

from openmdao.api import IndepVarComp, Component, Problem, Group

import numpy as np

# -------------------------------------------------------------------

class fin_csm(object):

    def __init__(self, fixed_charge_rate = 0.12, construction_finance_rate=0.0, tax_rate = 0.4, discount_rate = 0.07, \
                      construction_time = 1.0, project_lifetime = 20.0, sea_depth = 20.0):
        """
        OpenMDAO component to wrap finance model of the NREL _cost and Scaling Model (csmFinance.py)

        """

        super(fin_csm,self).__init__()

        #Outputs
        self.coe = 0.0 #Float(iotype='out', desc='Levelized cost of energy for the wind plant')
        self.lcoe = 0.0 #Float(iotype='out', desc='_cost of energy - unlevelized')

        # parameters
        self.fixed_charge_rate = fixed_charge_rate #Float(0.12, iotype = 'in', desc = 'fixed charge rate for coe calculation')
        self.construction_finance_rate = construction_finance_rate #Float(0.00, iotype='in', desc = 'construction financing rate applied to overnight capital costs')
        self.tax_rate = tax_rate #Float(0.4, iotype = 'in', desc = 'tax rate applied to operations')
        self.discount_rate = discount_rate #Float(0.07, iotype = 'in', desc = 'applicable project discount rate')
        self.construction_time = construction_time #Float(1.0, iotype = 'in', desc = 'number of years to complete project construction')
        self.project_lifetime = project_lifetime #Float(20.0, iotype = 'in', desc = 'project lifetime for LCOE calculation')
        self.sea_depth = sea_depth #Float(20.0, iotype='in', units='m', desc = 'depth of project for offshore, (0 for onshore)')

    def compute(self, turbine_cost, turbine_number, bos_costs, avg_annual_opex, net_aep):
        """
        Executes finance model of the NREL _cost and Scaling model to determine overall plant COE and LCOE.
        """

        # Inputs
        self.turbine_cost = turbine_cost #Float(iotype='in', desc = 'A Wind Turbine Capital _cost')
        self.turbine_number = turbine_number #Int(iotype = 'in', desc = 'number of turbines at plant')
        self.bos_costs = bos_costs #Float(iotype='in', desc='A Wind Plant Balance of Station _cost Model')
        self.avg_annual_opex = avg_annual_opex #Float(iotype='in', desc='A Wind Plant Operations Expenditures Model')
        self.net_aep = net_aep #Float(iotype='in', desc='A Wind Plant Annual Energy Production Model', units='kW*h')

        if self.sea_depth > 0.0:
           offshore = True
        else:
           offshore = False

        if offshore:
           warrantyPremium = (self.turbine_cost * self.turbine_number / 1.10) * 0.15
           icc = self.turbine_cost * self.turbine_number + warrantyPremium + self.bos_costs
        else:
           icc = self.turbine_cost * self.turbine_number + self.bos_costs

        #compute COE and LCOE values
        self.coe = (icc* self.fixed_charge_rate / self.net_aep) + \
                   (self.avg_annual_opex) * (1-self.tax_rate) / self.net_aep

        amortFactor = (1 + 0.5*((1+self.discount_rate)**self.construction_time-1)) * \
                      (self.discount_rate/(1-(1+self.discount_rate)**(-1.0*self.project_lifetime)))
        self.lcoe = (icc * amortFactor + self.avg_annual_opex)/self.net_aep


        # derivatives
        if offshore:
            self.d_coe_d_turbine_cost = (self.turbine_number * (1 + 0.15/1.10) * self.fixed_charge_rate) / self.net_aep
        else:
            self.d_coe_d_turbine_cost = self.turbine_number * self.fixed_charge_rate / self.net_aep
        self.d_coe_d_bos_cost = self.fixed_charge_rate / self.net_aep
        self.d_coe_d_avg_opex = (1-self.tax_rate) / self.net_aep
        self.d_coe_d_net_aep = -(icc * self.fixed_charge_rate + self.avg_annual_opex * (1-self.tax_rate)) / (self.net_aep**2)

        if offshore:
            self.d_lcoe_d_turbine_cost = self.turbine_number * (1 + 0.15/1.10) * amortFactor / self.net_aep
        else:
            self.d_lcoe_d_turbine_cost = self.turbine_number * amortFactor / self.net_aep
        self.d_lcoe_d_bos_cost = amortFactor / self.net_aep
        self.d_lcoe_d_avg_opex = 1. / self.net_aep
        self.d_lcoe_d_net_aep = -(icc * amortFactor + self.avg_annual_opex) / (self.net_aep**2)

    def list_deriv_vars(self):

        inputs = ['turbine_cost', 'bos_costs', 'avg_annual_opex', 'net_aep']
        outputs = ['coe', 'lcoe']

        return inputs, outputs

    def provideJ(self):

        # Jacobian
        self.J = np.array([[self.d_coe_d_turbine_cost, self.d_coe_d_bos_cost, self.d_coe_d_avg_opex, self.d_coe_d_net_aep],
                           [self.d_lcoe_d_turbine_cost, self.d_lcoe_d_bos_cost, self.d_lcoe_d_avg_opex, self.d_lcoe_d_net_aep]])

        return self.J


### FUSED-wrapper file (in WISDEM/Plant_CostsSE)
class fin_csm_fused(FUSED_Object):

    def __init__(self,fixed_charge_rate = 0.12, construction_finance_rate=0.0, tax_rate = 0.4, discount_rate = 0.07, \
                      construction_time = 1.0, project_lifetime = 20.0, sea_depth = 20.0):

        super(fin_csm_fused, self).__init__()

        self.implement_fifc(fifc_finance) # pulls in variables from fused-wind interface (not explicit)
        self.add_output(**fusedvar('lcoe',0.0)) 
        
        self.fin = fin_csm(fixed_charge_rate, construction_finance_rate, tax_rate, discount_rate, \
                      construction_time, project_lifetime, sea_depth)

    def compute(self, inputs, outputs):

        turbine_cost = inputs['turbine_cost']
        turbine_number = inputs['turbine_number']
        bos_costs = inputs['bos_costs']
        avg_annual_opex = inputs['avg_annual_opex']
        net_aep = inputs['net_aep']

        self.fin.compute(turbine_cost, turbine_number, bos_costs, avg_annual_opex, net_aep)

        # Outputs
        outputs['coe'] = self.fin.coe 
        outputs['lcoe'] = self.fin.lcoe 

        return outputs

def example_finance():

    # simple test of module

    # openmdao example of execution
    root = Group()
    root.add('fin_csm_test', FUSED_OpenMDAO(fin_csm_fused(fixed_charge_rate = 0.12, construction_finance_rate=0.0, tax_rate = 0.4, discount_rate = 0.07, \
                      construction_time = 1.0, project_lifetime = 20.0, sea_depth = 20.0)), promotes=['*'])
    prob = Problem(root)
    prob.setup()

    prob['turbine_cost'] = 6087803.555 / 50
    prob['turbine_number'] = 50
    preventative_opex = 401819.023
    lease_opex = 22225.395
    corrective_opex = 91048.387
    prob['avg_annual_opex'] = preventative_opex + corrective_opex + lease_opex
    prob['bos_costs'] = 7668775.3
    prob['net_aep'] = 15756299.843

    prob.run()
    print("Overall cost of energy for an offshore wind plant with 100 NREL 5 MW turbines")
    for io in root.unknowns:
        print(io + ' ' + str(root.unknowns[io]))

if __name__ == "__main__":

    example_finance()