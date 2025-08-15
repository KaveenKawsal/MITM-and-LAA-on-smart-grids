import matplotlib.pyplot as plt
from smart_grid_39bus import SmartGrid39Bus

def main():
    """
    Main function to run the smart grid simulation for the IEEE 39-bus system.
    """
    smart_grid = SmartGrid39Bus()

    # 1. Add distributed generation (with capacity factors for initial PG)
    dg_buses = [2, 10, 18, 25, 34]
    dg_capacities = [20, 35, 25, 40, 30]  # MW
    dg_types = ['solar', 'wind', 'biomass', 'solar', 'wind']
    dg_capacity_factors = [0.35, 0.45, 0.80, 0.30, 0.50]
    smart_grid.add_distributed_generation(dg_buses, dg_capacities, dg_types, dg_capacity_factors)

    # 2. Add energy storage
    storage_buses = [5, 15, 22, 30]
    storage_capacities = [15, 20, 25, 20]  # MW
    storage_durations = [4, 2, 6, 4]       # hours
    smart_grid.add_energy_storage(storage_buses, storage_capacities, storage_durations)

    # 3. Add EV charging BEFORE DR so DR baselines include EV load
    ev_buses = [4, 11, 23, 31, 38]
    ev_capacities = [150, 350, 150, 350, 150]  # kW per charger
    ev_counts = [10, 5, 8, 6, 12]
    smart_grid.add_ev_charging(ev_buses, ev_capacities, ev_counts, pf=0.95, utilization=0.30)

    # 4. Setup demand response (after EV load)
    dr_buses = [8, 12, 20, 28, 32]
    dr_capacities = [10, 15, 12, 18, 20]  # MW
    dr_types = ['price-based', 'incentive-based', 'price-based', 'incentive-based', 'price-based']
    smart_grid.setup_demand_response(dr_buses, dr_capacities, dr_types)

    # 5. Place PMUs
    pmu_buses = [1, 9, 19, 29, 39]
    smart_grid.place_pmus(pmu_buses)

    # 6. Base case PF
    print("\nRunning base case power flow...")
    smart_grid.run_power_flow()
    smart_grid.display_metrics()

    # 7. Activate demand response
    print("\nActivating demand response...")
    smart_grid.activate_demand_response(event_level='medium')

    print("\nRunning power flow with demand response...")
    smart_grid.run_power_flow()
    smart_grid.display_metrics()

    # 8. Optimize generation dispatch (runs PF internally)
    print("\nOptimizing generation dispatch...")
    smart_grid.optimize_dispatch()
    smart_grid.display_metrics()

    # 9. Visualization
    print("\nVisualizing network...")
    plt_obj = smart_grid.visualize_network(highlight_buses=dg_buses)
    plt_obj.savefig('ieee39_smartgrid.png')
    print("Network visualization saved as 'ieee39_smartgrid.png'")

    # 10. Contingency simulation
    print("\nSimulating contingency (branch index 5)...")
    smart_grid.simulate_contingency(branch_index=5)
    smart_grid.display_metrics()

    print("\nSmart grid simulation for IEEE 39-bus system is complete.")

if __name__ == "__main__":
    main()