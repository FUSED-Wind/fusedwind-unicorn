from fused_wind import create_interface, set_output, set_input

### Wind IO content (in windio - as yaml and in FUSED-Wind (or windio) as python dictionary translation)
# plant costs
bos_costs =  { 'name': 'bos_costs' , 'type': int, 'val': 1 }

# plant cost model description (basic)
machine_rating =  { 'name': 'machine_rating' , 'type': int, 'val': 1 }
rotor_diameter =  { 'name': 'rotor_diameter' , 'type': int, 'val': 1 }
hub_height =  { 'name': 'hub_height' , 'type': int, 'val': 1 }
RNA_mass =  { 'name': 'RNA_mass' , 'type': int, 'val': 1 }
turbine_cost =  { 'name': 'turbine_cost' , 'type': int, 'val': 1 }
turbine_number =  { 'name': 'turbine_number' , 'type': int, 'val': 1 }

### FUSED-interface content (in FUSED-Wind)
# bos_costs
fifc_bos_costs = create_interface()
set_output(fifc_bos_costs, bos_costs)
set_input(fifc_bos_costs, machine_rating)
set_input(fifc_bos_costs, rotor_diameter)
set_input(fifc_bos_costs, hub_height)
set_input(fifc_bos_costs, RNA_mass)
set_input(fifc_bos_costs, turbine_cost)
set_input(fifc_bos_costs, turbine_number)

## Turbine Cost
# turbine costs
turbine_cost = {'name': 'turbine_cost', 'type': float, 'val': 0.0}

# turbine cost model description (basic)
machine_rating =  { 'name': 'machine_rating' , 'type': float, 'val': 0.0 }
rotor_diameter =  { 'name': 'rotor_diameter' , 'type': float, 'val': 0.0 }
hub_height =  { 'name': 'hub_height' , 'type': float, 'val': 0.0 }
blade_number = { 'name': 'blade_number', 'type': int, 'val': 3}

# turbine cost 
fifc_tcc_costs = create_interface()
set_output(fifc_tcc_costs, turbine_cost)
set_input(fifc_tcc_costs, machine_rating)
set_input(fifc_tcc_costs, rotor_diameter)
set_input(fifc_tcc_costs, hub_height)
set_input(fifc_tcc_costs, blade_number)

## Financing
# financial model output
coe = {'name' : 'coe', 'type': float, 'val': 0.0}

# finance model description (basic)
turbine_cost = {'name': 'turbine_cost', 'type': float, 'val': 0.0}
turbine_number = {'name':'turbine_number', 'type':float, 'val': 0.0}
bos_costs = {'name' : 'bos_costs', 'type':float, 'val': 0.0}
avg_annual_opex = {'name':'avg_annual_opex', 'type':float, 'val': 0.0}
net_aep = {'name':'net_aep', 'type': float, 'val': 0.0}

# turbine cost 
fifc_finance = create_interface()
set_input(fifc_finance, turbine_cost)
set_input(fifc_finance, turbine_number)
set_input(fifc_finance, bos_costs)
set_input(fifc_finance, avg_annual_opex)
set_input(fifc_finance, net_aep)
set_output(fifc_finance, coe)