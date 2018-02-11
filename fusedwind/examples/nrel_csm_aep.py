"""
aero_csm_component.py

Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from math import pi, gamma, exp

from utilities import smooth_abs, smooth_min, hstack

from openmdao.api import IndepVarComp, Component, Problem, Group
from fused_wind import create_interface , FUSED_Object , FUSED_OpenMDAO , set_output, set_input, fusedvar


class aero_csm(object):

    def __init__(self):

        # Variables
        # machine_rating = Float(units = 'kW', iotype='in', desc= 'rated machine power in kW')
        # max_tip_speed = Float(units = 'm/s', iotype='in', desc= 'maximum allowable tip speed for the rotor')
        # rotor_diameter = Float(units = 'm', iotype='in', desc= 'rotor diameter of the machine')
        # max_power_coefficient = Float(iotype='in', desc= 'maximum power coefficient of rotor for operation in region 2')
        # opt_tsr = Float(iotype='in', desc= 'optimum tip speed ratio for operation in region 2')
        # cut_in_wind_speed = Float(units = 'm/s', iotype='in', desc= 'cut in wind speed for the wind turbine')
        # cut_out_wind_speed = Float(units = 'm/s', iotype='in', desc= 'cut out wind speed for the wind turbine')
        # hub_height = Float(units = 'm', iotype='in', desc= 'hub height of wind turbine above ground / sea level')
        # altitude = Float(units = 'm', iotype='in', desc= 'altitude of wind plant')
        # air_density = Float(units = 'kg / (m * m * m)', iotype='in', desc= 'air density at wind plant site')  # default air density value is 0.0 - forces aero csm to calculate air density in model
        # max_efficiency = Float(iotype='in', desc = 'maximum efficiency of rotor and drivetrain - at rated power')
        # thrust_coefficient = Float(iotype='in', desc='thrust coefficient at rated power')

        # Outputs
        self.rated_wind_speed = 0. # Float(units = 'm / s', iotype='out', desc='wind speed for rated power')
        self.rated_rotor_speed = 0. # Float(units = 'rpm', iotype='out', desc = 'rotor speed at rated power')
        self.rotor_thrust = 0. # Float(iotype='out', units='N', desc='maximum thrust from rotor')
        self.rotor_torque = 0. # Float(iotype='out', units='N * m', desc = 'torque from rotor at rated power')
        self.power_curve = np.zeros(161) # Array(iotype='out', units='kW', desc='total power before drivetrain losses')
        self.wind_curve = np.zeros(161) # Array(iotype='out', units='m/s', desc='wind curve associated with power curve')

    def compute(self, machine_rating, max_tip_speed, rotor_diameter, max_power_coefficient, opt_tsr,
                cut_in_wind_speed, cut_out_wind_speed, hub_height, altitude, air_density, max_efficiency,
                thrust_coefficient):
        """
        Executes Aerodynamics Sub-module of the NREL _cost and Scaling Model to create a power curve based on a limited set of inputs.
        It then modifies the ideal power curve to take into account drivetrain efficiency losses through an interface to a drivetrain efficiency model.
        """

        # initialize input parameters
        self.hubHt      = hub_height
        self.ratedPower = machine_rating
        self.maxTipSpd  = max_tip_speed
        self.rotorDiam  = rotor_diameter
        self.maxCp      = max_power_coefficient
        self.maxTipSpdRatio = opt_tsr
        self.cutInWS    =  cut_in_wind_speed
        self.cutOutWS   = cut_out_wind_speed

        if air_density == 0.0:
            # Compute air density
            ssl_pa     = 101300  # std sea-level pressure in Pa
            gas_const  = 287.15  # gas constant for air in J/kg/K
            gravity    = 9.80665 # standard gravity in m/sec/sec
            lapse_rate = 0.0065  # temp lapse rate in K/m
            ssl_temp   = 288.15  # std sea-level temp in K

            air_density = (ssl_pa * (1-((lapse_rate*(altitude + self.hubHt))/ssl_temp))**(gravity/(lapse_rate*gas_const))) / \
              (gas_const*(ssl_temp-lapse_rate*(altitude + self.hubHt)))
        else:
        		air_density = air_density

        # determine power curve inputs
        self.reg2pt5slope  = 0.05

        #self.max_efficiency = self.drivetrain.getMaxEfficiency()
        self.ratedHubPower = self.ratedPower / max_efficiency  # RatedHubPower

        self.omegaM = self.maxTipSpd/(self.rotorDiam/2.)  # Omega M - rated rotor speed
        omega0 = self.omegaM/(1+self.reg2pt5slope)       # Omega 0 - rotor speed at which region 2 hits zero torque
        Tm = self.ratedHubPower*1000/self.omegaM         # Tm - rated torque

        # compute rated rotor speed
        self.ratedRPM = (30./pi) * self.omegaM

        # compute variable-speed torque constant k
        kTorque = (air_density*pi*self.rotorDiam**5*self.maxCp)/(64*self.maxTipSpdRatio**3) # k

        b = -Tm/(self.omegaM-omega0)                       # b - quadratic formula values to determine omegaT
        c = (Tm*omega0)/(self.omegaM-omega0)               # c

        # omegaT is rotor speed at which regions 2 and 2.5 intersect
        # add check for feasibility of omegaT calculation 09/20/2012
        omegaTflag = True
        if (b**2-4*kTorque*c) > 0:
           omegaT = -(b/(2*kTorque))-(np.sqrt(b**2-4*kTorque*c)/(2*kTorque))  # Omega T

           windOmegaT = (omegaT*self.rotorDiam)/(2*self.maxTipSpdRatio) # Wind  at omegaT (M25)
           pwrOmegaT  = kTorque*omegaT**3/1000                                # Power at ometaT (M26)

        else:
           omegaTflag = False
           windOmegaT = self.ratedRPM
           pwrOmegaT = self.ratedPower

        # compute rated wind speed
        d = air_density*np.pi*self.rotorDiam**2.*0.25*self.maxCp
        self.ratedWindSpeed = \
           0.33*( (2.*self.ratedHubPower*1000.      / (    d))**(1./3.) ) + \
           0.67*( (((self.ratedHubPower-pwrOmegaT)*1000.) / (1.5*d*windOmegaT**2.))  + windOmegaT )

        # set up for idealized power curve
        n = 161 # number of wind speed bins
        itp = [None] * n
        ws_inc = 0.25  # size of wind speed bins for integrating power curve
        Wind = []
        Wval = 0.0
        Wind.append(Wval)
        for i in range(1,n):
           Wval += ws_inc
           Wind.append(Wval)

        # determine idealized power curve
        self.idealPowerCurve(Wind, itp, kTorque, windOmegaT, pwrOmegaT, n , omegaTflag)

        # add a fix for rated wind speed calculation inaccuracies kld 9/21/2012
        ratedWSflag = False
        # determine power curve after losses
        mtp = [None] * n
        for i in range(0,n):
           mtp[i] = itp[i] #* self.drivetrain.getDrivetrainEfficiency(itp[i],self.ratedHubPower)
           #print [Wind[i],itp[i],self.drivetrain.getDrivetrainEfficiency(itp[i],self.ratedHubPower),mtp[i]] # for testing
           if (mtp[i] > self.ratedPower):
              if not ratedWSflag:
                ratedWSflag = True
              mtp[i] = self.ratedPower

        self.rated_wind_speed = self.ratedWindSpeed
        self.rated_rotor_speed = self.ratedRPM
        self.power_curve = np.array(mtp)
        self.wind_curve = Wind

        # compute turbine load outputs
        self.rotor_torque = self.ratedHubPower/(self.ratedRPM*(pi/30.))*1000.
        self.rotor_thrust  = air_density * thrust_coefficient * pi * rotor_diameter**2 * (self.ratedWindSpeed**2) / 8.

    def idealPowerCurve( self, Wind, ITP, kTorque, windOmegaT, pwrOmegaT, n , omegaTflag):
        """
        Determine the ITP (idealized turbine power) array
        """

        idealPwr = 0.0

        for i in range(0,n):
            if (Wind[i] >= self.cutOutWS ) or (Wind[i] <= self.cutInWS):
                idealPwr = 0.0  # cut out
            else:
                if omegaTflag:
                    if ( Wind[i] > windOmegaT ):
                       idealPwr = (self.ratedHubPower-pwrOmegaT)/(self.ratedWindSpeed-windOmegaT) * (Wind[i]-windOmegaT) + pwrOmegaT # region 2.5
                    else:
                       idealPwr = kTorque * (Wind[i]*self.maxTipSpdRatio/(self.rotorDiam/2.0))**3 / 1000.0 # region 2
                else:
                    idealPwr = kTorque * (Wind[i]*self.maxTipSpdRatio/(self.rotorDiam/2.0))**3 / 1000.0 # region 2

            ITP[i] = idealPwr
            #print [Wind[i],ITP[i]]

        return

def weibull(X,K,L):
    '''
    Return Weibull probability at speed X for distribution with k=K, c=L

    Parameters
    ----------
    X : float
       wind speed of interest [m/s]
    K : float
       Weibull shape factor for site
    L : float
       Weibull scale factor for site [m/s]

    Returns
    -------
    w : float
      Weibull pdf value
    '''
    w = (K/L) * ((X/L)**(K-1)) * exp(-((X/L)**K))
    return w

class aep_csm(object):

    def __init__(self):

        # Variables
        # power_curve = Array(iotype='in', units='kW', desc='total power after drivetrain losses')
        # wind_curve = Array(iotype='in', units='m/s', desc='wind curve associated with power curve')
        # hub_height = Float(iotype='in', units = 'm', desc='hub height of wind turbine above ground / sea level')
        # shear_exponent = Float(iotype='in', desc= 'shear exponent for wind plant') #TODO - could use wind model here
        # wind_speed_50m = Float(iotype='in', units = 'm/s', desc='mean annual wind speed at 50 m height')
        # weibull_k= Float(iotype='in', desc = 'weibull shape factor for annual wind speed distribution')
        # machine_rating = Float(iotype='in', units='kW', desc='machine power rating')

        # Parameters
        # soiling_losses = Float(0.0, iotype='in', desc = 'energy losses due to blade soiling for the wind plant - average across turbines')
        # array_losses = Float(0.06, iotype='in', desc = 'energy losses due to turbine interactions - across entire plant')
        # availability = Float(0.94287630736, iotype='in', desc = 'average annual availbility of wind turbines at plant')
        # turbine_number = Int(100, iotype='in', desc = 'total number of wind turbines at the plant')

        # Output
        gross_aep = 0. # Float(iotype='out', desc='Gross Annual Energy Production before availability and loss impacts', unit='kWh')
        net_aep = 0. # Float(units= 'kW * h', iotype='out', desc='Annual energy production in kWh')  # use PhysicalUnits to set units='kWh'
        power_array = 0. # Array(iotype='out', units='kW', desc='total power after drivetrain losses')
        capacity_factor = 0. # Float(iotype='out', desc='plant capacity factor')

    def compute(self, power_curve, wind_curve, hub_height, shear_exponent,
                wind_speed_50m, weibull_k, machine_rating, soiling_losses,
                array_losses, availability, turbine_number):
        """
        Executes AEP Sub-module of the NREL _cost and Scaling Model by convolving a wind turbine power curve with a weibull distribution.
        It then discounts the resulting AEP for availability, plant and soiling losses.
        """

        power_array = np.array([wind_curve, power_curve])

        hubHeightWindSpeed = ((hub_height/50)**shear_exponent)*wind_speed_50m
        K = weibull_k
        L = hubHeightWindSpeed / exp(np.log(gamma(1.+1./K)))

        turbine_energy = 0.0
        for i in range(0,power_array.shape[1]):
           X = power_array[0,i]
           result = power_array[1,i] * weibull(X, K, L)
           turbine_energy += result

        ws_inc = power_array[0,1] - power_array[0,0]
        self.gross_aep = turbine_energy * 8760.0 * turbine_number * ws_inc
        self.net_aep = self.gross_aep * (1.0-soiling_losses)* (1.0-array_losses) * availability
        self.capacity_factor = self.net_aep / (8760 * machine_rating)

class drivetrain_csm(object):
    """drivetrain losses from NREL cost and scaling model"""

    def __init__(self, drivetrain_type='geared'):

        self.drivetrain_type = drivetrain_type

        power = np.zeros(161) # Array(iotype='out', units='kW', desc='total power after drivetrain losses')

    def compute(self, aero_power, aero_torque, aero_thrust, rated_power):


        if self.drivetrain_type == 'geared':
            constant = 0.01289
            linear = 0.08510
            quadratic = 0.0

        elif self.drivetrain_type == 'single_stage':
            constant = 0.01331
            linear = 0.03655
            quadratic = 0.06107

        elif self.drivetrain_type == 'multi_drive':
            constant = 0.01547
            linear = 0.04463
            quadratic = 0.05790

        elif self.drivetrain_type == 'pm_direct_drive':
            constant = 0.01007
            linear = 0.02000
            quadratic = 0.06899


        Pbar0 = aero_power / rated_power

        # handle negative power case (with absolute value)
        Pbar1, dPbar1_dPbar0 = smooth_abs(Pbar0, dx=0.01)

        # truncate idealized power curve for purposes of efficiency calculation
        Pbar, dPbar_dPbar1, _ = smooth_min(Pbar1, 1.0, pct_offset=0.01)

        # compute efficiency
        eff = 1.0 - (constant/Pbar + linear + quadratic*Pbar)

        self.power = aero_power * eff

    def provideJ(self):

        # gradients
        dPbar_dPa = dPbar_dPbar1*dPbar1_dPbar0/rated_power
        dPbar_dPr = -dPbar_dPbar1*dPbar1_dPbar0*aero_power/rated_power**2

        deff_dPa = dPbar_dPa*(constant/Pbar**2 - quadratic)
        deff_dPr = dPbar_dPr*(constant/Pbar**2 - quadratic)

        dP_dPa = eff + aero_power*deff_dPa
        dP_dPr = aero_power*deff_dPr

        self.J = hstack([np.diag(dP_dPa), dP_dPr])


        return self.J


class aep_csm_assembly(object):

    def __init__(self, drivetrain_type='geared'):
        self.aero = aero_csm()
        self.drivetrain = drivetrain_csm(drivetrain_type)
        self.aep = aep_csm()

    def compute(self, machine_rating, max_tip_speed, rotor_diameter, max_power_coefficient, opt_tsr,
                cut_in_wind_speed, cut_out_wind_speed, hub_height, altitude, air_density,
                max_efficiency, thrust_coefficient, soiling_losses, array_losses, availability,
                turbine_number, shear_exponent, wind_speed_50m, weibull_k):

        self.aero.compute(machine_rating, max_tip_speed, rotor_diameter, max_power_coefficient, opt_tsr,
                    cut_in_wind_speed, cut_out_wind_speed, hub_height, altitude, air_density, max_efficiency,
                    thrust_coefficient)

        self.drivetrain.compute(self.aero.power_curve, self.aero.rotor_torque, self.aero.rotor_thrust, machine_rating)

        self.aep.compute(self.drivetrain.power, self.aero.wind_curve, hub_height, shear_exponent, wind_speed_50m,
                        weibull_k, machine_rating, soiling_losses, array_losses, availability,
                        turbine_number)


class aep_csm_fused(FUSED_Object):

    def __init__(self):
        super(aep_csm_fused, self).__init__()

        # Add model specific inputs
        self.add_input(**fusedvar('machine_rating', 100.))
        self.add_input(**fusedvar('max_tip_speed',0.0))
        self.add_input(**fusedvar('rotor_diameter',0.0))
        self.add_input(**fusedvar('max_power_coefficient',0.0))
        self.add_input(**fusedvar('opt_tsr',0.0))
        self.add_input(**fusedvar('cut_in_wind_speed',0.0))
        self.add_input(**fusedvar('cut_out_wind_speed',0.0))
        self.add_input(**fusedvar('air_density',0.0))
        self.add_input(**fusedvar('max_efficiency',0.0))
        self.add_input(**fusedvar('thrust_coefficient',0.0))
        self.add_input(**fusedvar('soiling_losses',0.0))
        self.add_input(**fusedvar('array_losses',0.0))
        self.add_input(**fusedvar('availability',0.0))
        self.add_input(**fusedvar('turbine_number',0.0))
        self.add_input(**fusedvar('shear_exponent',0.0))
        self.add_input(**fusedvar('wind_speed_50m',0.0))
        self.add_input(**fusedvar('weibull_k',0.0))
        self.add_input(**fusedvar('hub_height',0.0))
        self.add_input(**fusedvar('altitude',0.0))

        # Add model specific outputs
        self.add_output(**fusedvar('rated_wind_speed', 0.)) # = Float(11.506, units = 'm / s', iotype='out', desc='wind speed for rated power')
        self.add_output(**fusedvar('rated_rotor_speed', 0.)) # = Float(12.126, units = 'rpm', iotype='out', desc = 'rotor speed at rated power')
        self.add_output(**fusedvar('rotor_thrust', 0.)) # = Float(iotype='out', units='N', desc='maximum thrust from rotor')
        self.add_output(**fusedvar('rotor_torque', 0.)) # = Float(iotype='out', units='N * m', desc = 'torque from rotor at rated power')
        # self.add_output(**fusedvar('power_curve', np.zeros(161))) # = Array(np.array([[4.0,80.0],[25.0, 5000.0]]), iotype='out', desc = 'power curve for a particular rotor')
        self.add_output(**fusedvar('gross_aep', 0.)) # = Float(0.0, iotype='out', desc='Gross Annual Energy Production before availability and loss impacts', unit='kWh')
        self.add_output(**fusedvar('net_aep', 0.)) # = Float(0.0, units= 'kW * h', iotype='out', desc='Annual energy production in kWh')  # use PhysicalUnits to set units='kWh'
        self.add_output(**fusedvar('capacity_factor', 0.)) # = Float(iotype='out', desc='plant capacity factor')

        self.aep_csm_assembly = aep_csm_assembly()


    def compute(self, inputs, outputs):
        self.aep_csm_assembly.compute(inputs['machine_rating'], inputs['max_tip_speed'], inputs['rotor_diameter'], inputs['max_power_coefficient'], inputs['opt_tsr'],
                inputs['cut_in_wind_speed'], inputs['cut_out_wind_speed'], inputs['hub_height'], inputs['altitude'], inputs['air_density'],
                inputs['max_efficiency'], inputs['thrust_coefficient'], inputs['soiling_losses'], inputs['array_losses'], inputs['availability'],
                inputs['turbine_number'], inputs['shear_exponent'], inputs['wind_speed_50m'], inputs['weibull_k'])

        outputs['rated_wind_speed'] = self.aep_csm_assembly.aero.rated_wind_speed
        outputs['rated_rotor_speed'] = self.aep_csm_assembly.aero.rated_rotor_speed
        outputs['rotor_thrust'] = self.aep_csm_assembly.aero.rotor_thrust
        outputs['rotor_torque'] = self.aep_csm_assembly.aero.rotor_torque
        # outputs['power_curve'] = self.aep_csm_assembly.aero.power_curve
        outputs['gross_aep'] = self.aep_csm_assembly.aep.gross_aep
        outputs['net_aep'] = self.aep_csm_assembly.aep.net_aep
        outputs['capacity_factor'] = self.aep_csm_assembly.aep.capacity_factor

def example():

    root = Group()
    root.add('aep_test', FUSED_OpenMDAO(aep_csm_fused()), promotes=['*'])

    prob = Problem(root)
    prob.setup()

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

    prob.run()

    print("AEP output")
    for io in root.unknowns:
        print(io + ' ' + str(root.unknowns[io]))

if __name__=="__main__":


    example()