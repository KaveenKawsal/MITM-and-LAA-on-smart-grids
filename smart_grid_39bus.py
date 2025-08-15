import numpy as np
import networkx as nx
from pypower.api import runpf, ppoption
from pypower.case39 import case39  # IEEE 39-bus test system

class SmartGrid39Bus:
    """
    IEEE 39-bus system with added smart grid features:
    - Distributed Generation (added as separate generators)
    - Energy Storage (capacity tracking; injection placeholder)
    - Demand Response (load reduction with baseline)
    - EV Charging Load (adds P & Q with assumed PF)
    - PMU placement
    - Metrics & simple dispatch
    """

    def __init__(self):
        self.case_data = case39()

        # Original snapshots
        self.original_bus_data = self.case_data['bus'].copy()
        self.original_gen_data = self.case_data['gen'].copy()
        self.original_branch_data = self.case_data['branch'].copy()
        self.original_gencost = self.case_data['gencost'].copy()

        # Smart grid components
        self.energy_storage = {}          # bus_id -> dict
        self.distributed_gen = {}         # key -> dict
        self.demand_response = {}         # bus_id -> dict (baseline, max_reduction)
        self.pmu_locations = []
        self.ev_charging = {}             # bus_id -> dict

        # Results
        self.last_pf_results = None
        self.last_pf_success = False
        self.network_graph = None

        # Metrics
        self.metrics = {
            'reliability_index': 0.0,
            'voltage_uniformity': 0.0,
            'power_losses_MW': 0.0,
            'renewable_penetration_capacity_pct': 0.0,
            'renewable_penetration_output_pct': 0.0,
            'demand_response_participation_MW': 0.0
        }

    # -------------------- Helpers --------------------

    def _bus_index(self, bus_id):
        # Assumes bus numbering contiguous (valid for case39)
        return int(bus_id) - 1

    def _is_renewable(self, typ):
        return typ.lower() in ('solar', 'wind', 'hydro', 'biomass')

    # -------------------- DG --------------------

    def add_distributed_generation(self, bus_ids, capacities, types, capacity_factors=None):
        """
        Add DG units as separate generator rows (do not inflate existing Pmax).
        capacity_factors: optional list (0..1) for initial PG dispatch; default 0.3.
        """
        if not (len(bus_ids) == len(capacities) == len(types)):
            raise ValueError("DG lists must have equal length")
        if capacity_factors is None:
            capacity_factors = [0.3] * len(bus_ids)
        if len(capacity_factors) != len(bus_ids):
            raise ValueError("capacity_factors length mismatch")

        gen_table = self.case_data['gen']
        gencost = self.case_data['gencost']
        base_rows = gen_table.shape[0]

        new_gen_rows = []
        new_cost_rows = []
        for i, bus_id in enumerate(bus_ids):
            cap = float(capacities[i])
            if cap <= 0:
                raise ValueError(f"DG capacity must be >0 (bus {bus_id})")
            cf = max(0.0, min(1.0, capacity_factors[i]))
            pg0 = cap * cf
            qmax = 0.5 * cap
            qmin = -0.5 * cap
            # MATPOWER gen format indices:
            # 0: BUS, 1: PG, 2: QG, 3: QMAX, 4: QMIN, 5: VG, 6: MBASE, 7: STATUS,
            # 8: PMAX, 9: PMIN, remaining cost-related placeholders -> fill zeros
            row = np.zeros(21)
            row[0] = bus_id
            row[1] = pg0
            row[2] = 0.0
            row[3] = qmax
            row[4] = qmin
            row[5] = 1.0
            row[6] = 100.0
            row[7] = 1
            row[8] = cap
            row[9] = 0.0
            new_gen_rows.append(row)

            # Simple quadratic cost: a*P^2 + b*P + c (make renewables low marginal)
            # gencost format: 2 startup shutdown n c(n-1) ... c0
            if self._is_renewable(types[i]):
                cost = [2, 0, 0, 3, 0.0001, 5.0, 0.0]
            else:
                cost = [2, 0, 0, 3, 0.001, 10.0, 0.0]
            new_cost_rows.append(cost)

            self.distributed_gen[f"DG_{base_rows + len(new_gen_rows) - 1}"] = {
                'bus': bus_id,
                'pmax': cap,
                'type': types[i],
                'capacity_factor': cf
            }

        if new_gen_rows:
            self.case_data['gen'] = np.vstack([gen_table, np.array(new_gen_rows)])
            self.case_data['gencost'] = np.vstack([gencost, np.array(new_cost_rows)])

        print(f"Added {len(new_gen_rows)} DG units (separate generators)")
        # Metrics will update after next power flow for voltage/losses; still update capacity metrics
        self.update_metrics(voltage_loss_from_pf=False)

    # -------------------- Storage --------------------

    def add_energy_storage(self, bus_ids, capacities, durations):
        if not (len(bus_ids) == len(capacities) == len(durations)):
            raise ValueError("Storage lists must have equal length")
        for i, bus_id in enumerate(bus_ids):
            cap = float(capacities[i])
            dur = float(durations[i])
            if cap <= 0 or dur <= 0:
                raise ValueError("Storage capacity/duration must be >0")
            self.energy_storage[bus_id] = {
                'power_MW': cap,
                'duration_h': dur,
                'energy_MWh': cap * dur,
                'soc_MWh': 0.5 * cap * dur,
                'status': 'idle'
            }
        print(f"Added {len(bus_ids)} energy storage systems")
        self.update_metrics(voltage_loss_from_pf=False)
    def setup_demand_response(self, bus_ids, dr_capacities_MW, dr_types):
        if not (len(bus_ids) == len(dr_capacities_MW) == len(dr_types)):
            raise ValueError("DR lists must have equal length")
        bus = self.case_data['bus']
        for i, bus_id in enumerate(bus_ids):
            idx = self._bus_index(bus_id)
            pd_now = bus[idx, 2]
            qd_now = bus[idx, 3]
            max_red = min(float(dr_capacities_MW[i]), pd_now)  # cannot exceed current load
            self.demand_response[bus_id] = {
                'baseline_P': pd_now,
                'baseline_Q': qd_now,
                'max_reduction_P': max_red,
                'type': dr_types[i],
                'activated_P': 0.0
            }
        print(f"Configured DR programs on {len(bus_ids)} buses")
        self.update_metrics(voltage_loss_from_pf=False)
    def place_pmus(self, bus_ids):
        self.pmu_locations = list(bus_ids)
        observability = len(bus_ids) / self.case_data['bus'].shape[0]
        print(f"Placed {len(bus_ids)} PMUs (observability {observability:.1%})")

    def add_ev_charging(self, bus_ids, capacities_kw, charger_counts, pf=0.95, utilization=0.3):
        if not (len(bus_ids) == len(capacities_kw) == len(charger_counts)):
            raise ValueError("EV lists must have equal length")
        if not (0 < pf <= 1):
            raise ValueError("Power factor must be (0,1]")
        tan_phi = np.tan(np.arccos(pf))
        bus = self.case_data['bus']
        for i, bus_id in enumerate(bus_ids):
            p_per = capacities_kw[i] / 1000.0  # kW -> MW per charger
            count = charger_counts[i]
            p_add = p_per * count * utilization
            q_add = p_add * tan_phi
            idx = self._bus_index(bus_id)
            bus[idx, 2] += p_add      # PD
            bus[idx, 3] += q_add      # QD
            self.ev_charging[bus_id] = {
                'p_installed_MW': p_per * count,
                'p_active_MW': p_add,
                'q_active_MVAr': q_add,
                'pf': pf,
                'utilization': utilization
            }
        print(f"Added EV charging on {len(bus_ids)} buses (utilization {utilization*100:.0f}%, pf={pf})")
        self.update_metrics(voltage_loss_from_pf=False)

    def run_power_flow(self, quiet=True):
        ppopt = ppoption(PF_ALG=1, VERBOSE=0 if quiet else 1, OUT_ALL=0)
        results, success = runpf(self.case_data, ppopt)
        self.last_pf_results = results if success else None
        self.last_pf_success = bool(success)
        if success:
            print("Power flow converged")
            # Update bus & gen states to internal case (pypower returns updated copies)
            self.case_data['bus'] = results['bus']
            self.case_data['gen'] = results['gen']
            self.case_data['branch'] = results['branch']
            self.update_metrics()
        else:
            print("Power flow did not converge")
        return success
    def activate_demand_response(self, event_level='medium'):
        if not self.demand_response:
            print("No DR configured")
            return
        factor_map = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        f = factor_map.get(event_level, 0.6)
        bus = self.case_data['bus']
        for bus_id, info in self.demand_response.items():
            idx = self._bus_index(bus_id)
            target = info['max_reduction_P'] * f
            # Already activated portion?
            remaining = info['max_reduction_P'] - info['activated_P']
            apply_red = min(target, remaining)
            if apply_red <= 0:
                continue
            # Scale Q with same proportion relative to baseline
            if info['baseline_P'] > 0:
                q_ratio = apply_red / info['baseline_P']
            else:
                q_ratio = 0.0
            q_red = info['baseline_Q'] * q_ratio
            # Apply
            bus[idx, 2] -= apply_red
            bus[idx, 3] -= q_red
            bus[idx, 2] = max(bus[idx, 2], 0.0)
            bus[idx, 3] = max(bus[idx, 3], 0.0)
            info['activated_P'] += apply_red
        print(f"DR activated at level '{event_level}' (factor {f})")
        self.update_metrics(voltage_loss_from_pf=False)
    def visualize_network(self, highlight_buses=None):
        from matplotlib import pyplot as plt
        if highlight_buses is None:
            highlight_buses = []
        if self.network_graph is None:
            G = nx.Graph()
            branch = self.case_data['branch']
            for row in branch:
                if int(row[10]) == 0:  # out-of-service
                    continue
                f, t = int(row[0]), int(row[1])
                G.add_edge(f, t)
            self.network_graph = G
        pos = nx.spring_layout(self.network_graph, seed=39)
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_edges(self.network_graph, pos, alpha=0.4)
        node_colors = []
        for n in self.network_graph.nodes():
            if n in highlight_buses:
                node_colors.append('gold')
            elif n in self.pmu_locations:
                node_colors.append('cyan')
            else:
                node_colors.append('lightgray')
        nx.draw_networkx_nodes(self.network_graph, pos, node_color=node_colors, node_size=400, edgecolors='k')
        nx.draw_networkx_labels(self.network_graph, pos, font_size=8)
        plt.title("IEEE 39-Bus Smart Grid Topology")
        plt.axis('off')
        return plt
    def update_metrics(self, voltage_loss_from_pf=True):
        gen = self.case_data['gen']
        bus = self.case_data['bus']
        total_capacity = np.sum(gen[:, 8])
        total_pg = np.sum(gen[:, 1])
        load_p = np.sum(bus[:, 2])

        # Renewable identification (by DG record)
        renewable_pmax = 0.0
        renewable_pg = 0.0
        for key, info in self.distributed_gen.items():
            if self._is_renewable(info['type']):
                renewable_pmax += info['pmax']
                # Find matching gen row by bus & pmax (approximate)
                # (Could store row index when adding)
        # Better: store row indices
        for key, info in self.distributed_gen.items():
            if '_row' in info:
                pass  # placeholder if expanded later

        # More robust: treat any DG types flagged as renewable by bus matching (may double count if multiple)
        # Simpler approach: iterate rows that were added (those beyond original_gen_data length)
        orig_rows = self.original_gen_data.shape[0]
        if gen.shape[0] > orig_rows:
            added = gen[orig_rows:]
            for i, (k, info) in enumerate(self.distributed_gen.items()):
                if self._is_renewable(info['type']) and i < added.shape[0]:
                    renewable_pg += added[i, 1]
        # Capacity from DG records
        renewable_pmax = sum(info['pmax'] for info in self.distributed_gen.values()
                             if self._is_renewable(info['type']))

        cap_pen = (renewable_pmax / total_capacity * 100) if total_capacity > 0 else 0.0
        out_pen = (renewable_pg / total_pg * 100) if total_pg > 0 else 0.0

        # Reliability (simple): reserve margin over load
        reserve = total_capacity - load_p
        reliability = 1 - np.exp(-max(reserve, 0.0) / max(load_p, 1e-6))

        # Voltage uniformity & losses only if we have PF results now (and requested)
        if voltage_loss_from_pf and self.last_pf_success and self.last_pf_results is not None:
            vmag = self.last_pf_results['bus'][:, 7]  # VM column
            vu = 1.0 / (1.0 + np.std(vmag))
            # Branch losses: Pf + Pt (should be ≥0). Columns 13 (PF), 15 (PT)
            branch = self.last_pf_results['branch']
            pf = branch[:, 13]
            pt = branch[:, 15]
            line_losses = pf + pt
            # Numerical noise clamp
            line_losses[line_losses < 0] = 0
            total_losses = np.sum(line_losses)
        else:
            vu = self.metrics.get('voltage_uniformity', 0.0)
            total_losses = self.metrics.get('power_losses_MW', 0.0)

        dr_participation = sum(info['activated_P'] for info in self.demand_response.values())

        self.metrics.update({
            'reliability_index': float(reliability),
            'voltage_uniformity': float(vu),
            'power_losses_MW': float(total_losses),
            'renewable_penetration_capacity_pct': float(cap_pen),
            'renewable_penetration_output_pct': float(out_pen),
            'demand_response_participation_MW': float(dr_participation)
        })

    def display_metrics(self):
        print("\n--- Smart Grid Metrics ---")
        for k, v in self.metrics.items():
            if 'pct' in k:
                print(f"{k}: {v:.2f}%")
            elif k.endswith('_MW'):
                print(f"{k}: {v:.2f} MW")
            else:
                print(f"{k}: {v:.4f}")
        print("--------------------------")

    def simulate_contingency(self, branch_index):
        branch = self.case_data['branch']
        if branch_index < 0 or branch_index >= branch.shape[0]:
            print("Invalid branch index")
            return
        original_status = branch[branch_index, 10]  # STATUS column
        branch[branch_index, 10] = 0
        print(f"Branch {branch_index} outaged")
        self.case_data['branch'] = branch
        success = self.run_power_flow()
        # Restore
        branch[branch_index, 10] = original_status
        self.case_data['branch'] = branch
        if success:
            print("Contingency analysis complete (branch restored)")
        else:
            print("Contingency PF failed")

    def optimize_dispatch(self):
        """
        Simple proportional re-dispatch of all online generators to meet load + losses.
        """
        if not self.last_pf_success:
            print("Run a converged power flow before dispatch")
            return
        gen = self.case_data['gen']
        bus = self.case_data['bus']
        total_load = np.sum(bus[:, 2])
        # Approximate losses from last metrics
        losses = max(0.0, self.metrics.get('power_losses_MW', 0.0))
        demand = total_load + losses

        pmax = gen[:, 8]
        pmin = gen[:, 9]
        headroom = pmax - pmin
        total_headroom = np.sum(headroom)
        if total_headroom <= 0:
            print("No dispatchable headroom")
            return
        # Allocate proportionally
        dispatch = pmin + (headroom * demand / total_headroom)
        # Cap to pmax
        dispatch = np.minimum(dispatch, pmax)
        gen[:, 1] = dispatch  # PG
        self.case_data['gen'] = gen
        print("Dispatch updated (proportional). Running PF...")
        self.run_power_flow()
    def reset_to_original(self):
        self.case_data['bus'] = self.original_bus_data.copy()
        self.case_data['gen'] = self.original_gen_data.copy()
        self.case_data['branch'] = self.original_branch_data.copy()
        self.case_data['gencost'] = self.original_gencost.copy()
        self.energy_storage.clear()
        self.distributed_gen.clear()
        self.demand_response.clear()
        self.ev_charging.clear()
        self.pmu_locations.clear()
        self.last_pf_results = None
        self.last_pf_success = False
        for k in self.metrics:
            self.metrics[k] = 0.0
        print("System reset to original IEEE 39-bus state")