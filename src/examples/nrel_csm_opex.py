"""
om_csm_component.py

Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""
import numpy as np

from openmdao.api import IndepVarComp, Component, Problem, Group

from fused_wind import create_interface , FUSED_Object , FUSED_OpenMDAO , set_output, set_input, fusedvar

from config import *

class opex_csm_fused(FUSED_Object):

    def __init__(self):
        super(opex_csm_fused, self).__init__()

        # Add model specific inputs
        self.add_input(**fusedvar('sea_depth',0.0)) # #20.0, units = 'm', iotype = 'in', desc = 'sea depth for offshore wind plant')
        self.add_input(**fusedvar('year',0.0)) # = Int(2009, iotype='in', desc='year for project start')
        self.add_input(**fusedvar('month',0.0)) # iotype = 'in', desc= 'month for project start') # units = months
        self.add_input(**fusedvar('turbine_number',0.0)) # iotype = 'in', desc = 'number of turbines at plant')
        self.add_input(**fusedvar('machine_rating',0.0)) # units = 'kW', iotype = 'in', desc = 'rated power for a wind turbine')
        self.add_input(**fusedvar('net_aep',0.0)) # units = 'kW * h', iotype = 'in', desc = 'annual energy production for the plant')

        # Add model specific outputs
        self.add_output(**fusedvar('avg_annual_opex',0.0)) # desc='Average annual Operating Expenditures for a wind plant over its lifetime')
        self.add_output(**fusedvar('opex_breakdown_preventative_opex',0.0)) # desc='annual expenditures on preventative maintenance - BOP and turbines'
        self.add_output(**fusedvar('opex_breakdown_corrective_opex',0.0)) # desc='annual unscheduled maintenance costs (replacements) - BOP and turbines'
        self.add_output(**fusedvar('opex_breakdown_lease_opex',0.0)) # desc='annual lease expenditures'
        self.add_output(**fusedvar('opex_breakdown_other_opex',0.0)) # desc='other operational expenditures such as fixed costs'

        self.opex = opex_csm_component()

    def compute(self, inputs, outputs):

        self.opex.compute(inputs['sea_depth'], inputs['year'], inputs['month'],
                          inputs['turbine_number'], inputs['machine_rating'], inputs['net_aep'])

        outputs['avg_annual_opex'] = self.opex.avg_annual_opex
        outputs['opex_breakdown_preventative_opex'] = self.opex.opex_breakdown_preventative_opex
        outputs['opex_breakdown_corrective_opex'] = self.opex.opex_breakdown_corrective_opex
        outputs['opex_breakdown_lease_opex'] = self.opex.opex_breakdown_lease_opex
        outputs['opex_breakdown_other_opex'] = self.opex.opex_breakdown_other_opex

class opex_csm_component(object):


    def __init__(self):
        # variables

        # Outputs
        self.avg_annual_opex = 0.

        # self.opex_breakdown = VarTree(OPEXVarTree(),iotype='out')
        self.opex_breakdown_preventative_opex = 0.
        self.opex_breakdown_corrective_opex = 0.
        self.opex_breakdown_lease_opex = 0.
        self.opex_breakdown_other_opex = 0.

    def compute(self, sea_depth, year, month, turbine_number, machine_rating, net_aep):

        # initialize variables
        if sea_depth == 0:
            offshore = False
        else:
            offshore = True
        ppi.curr_yr = year
        ppi.curr_mon = month

        #O&M
        offshoreCostFactor = 0.0200  # $/kwH
        landCostFactor     = 0.0070  # $/kwH
        if not offshore:  # kld - place for an error check - iShore should be in 1:4
            cost = net_aep * landCostFactor
            costEscalator = ppi.compute('IPPI_LOM')
        else:
            cost = net_aep * offshoreCostFactor
            ppi.ref_yr = 2003
            costEscalator = ppi.compute('IPPI_OOM')
            ppi.ref_yr = 2002

        self.opex_breakdown_preventative_opex = cost * costEscalator # in $/year

        #LRC
        if not offshore:
            lrcCF = 10.70 # land based
            costlrcEscFactor = ppi.compute('IPPI_LLR')
        else: #TODO: transition and deep water options if applicable
            lrcCF = 17.00 # offshore
            ppi.ref_yr = 2003
            costlrcEscFactor = ppi.compute('IPPI_OLR')
            ppi.ref_yr = 2002

        self.opex_breakdown_corrective_opex = machine_rating * lrcCF * costlrcEscFactor * turbine_number # in $/yr

        #LLC
        if not offshore:
            leaseCF = 0.00108 # land based
            costlandEscFactor = ppi.compute('IPPI_LSE')
        else: #TODO: transition and deep water options if applicable
            leaseCF = 0.00108 # offshore
            costlandEscFactor = ppi.compute('IPPI_LSE')

        self.opex_breakdown_lease_opex = net_aep * leaseCF * costlandEscFactor # in $/yr

        #Other
        self.opex_breakdown_other_opex = 0.0

        #Total OPEX
        self.avg_annual_opex = self.opex_breakdown_preventative_opex + self.opex_breakdown_corrective_opex \
           + self.opex_breakdown_lease_opex

    def compute_partials(self):

        #dervivatives
        self.d_corrective_d_aep = 0.0
        self.d_corrective_d_rating = lrcCF * costlrcEscFactor * self.turbine_number
        self.d_lease_d_aep = leaseCF * costlandEscFactor
        self.d_lease_d_rating = 0.0
        self.d_other_d_aep = 0.0
        self.d_other_d_rating = 0.0
        if not offshore:
            self.d_preventative_d_aep = landCostFactor * costEscalator
        else:
            self.d_preventative_d_aep = offshoreCostFactor * costEscalator
        self.d_preventative_d_rating = 0.0
        self.d_opex_d_aep = self.d_preventative_d_aep + self.d_corrective_d_aep + self.d_lease_d_aep + self.d_other_d_aep
        self.d_opex_d_rating = self.d_preventative_d_rating + self.d_corrective_d_rating + self.d_lease_d_rating + self.d_other_d_rating


        self.J = np.array([[self.d_preventative_d_aep, self.d_preventative_d_rating], [self.d_corrective_d_aep, self.d_corrective_d_rating],\
                           [self.d_lease_d_aep, self.d_lease_d_rating], [self.d_other_d_aep, self.d_other_d_rating],\
                           [self.d_opex_d_aep, self.d_opex_d_rating]])

        return self.J


def example():

    # simple test of module

    root = Group()
    root.add('bos_opex_test', FUSED_OpenMDAO(opex_csm_fused()), promotes=['*'])

    prob = Problem(root)
    prob.setup()

    prob['machine_rating'] = 5000.0 # Need to manipulate input or underlying cprob[onent will not execute
    prob['net_aep'] = 1701626526.28
    prob['sea_depth'] = 20.0
    prob['year'] = 2009
    prob['month'] = 12
    prob['turbine_number'] = 100

    prob.run()
    print("Average annual operational expenditures for an offshore wind plant with 100 NREL 5 MW turbines")
    for io in root.unknowns:
        print(io + ' ' + str(root.unknowns[io]))

    prob['sea_depth'] = 0.0
    prob.run()
    print("Average annual operational expenditures for an land-based wind plant with 100 NREL 5 MW turbines")
    for io in root.unknowns:
        print(io + ' ' + str(root.unknowns[io]))

if __name__ == "__main__":

    example()