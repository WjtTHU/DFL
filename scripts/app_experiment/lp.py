"""Continuous APP model, LP export, CPLEX CLI, and independent residual checks."""
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class AppLP:
    names: list
    objective: np.ndarray
    a_eq: np.ndarray
    b_eq: np.ndarray
    a_ub: np.ndarray
    b_ub: np.ndarray
    upper: np.ndarray
    indices: dict
    shape: tuple


def build_app_lp(demand, capacity, bom, opening_raw, opening_finished, opening_backlog,
                 primary_arrivals, committed_secondary, secondary_caps, lead, costs, terminal_penalty):
    """Primary arrivals DURING k only support production in k+1 and later.

    opening_raw includes secondary receipts due at this decision. committed_secondary[0]
    must therefore be zero. Future committed orders are constants, not new decisions.
    """
    demand = np.asarray(demand, dtype=float)
    n, products = demand.shape
    materials = len(opening_raw)
    bom = np.asarray(bom, dtype=float)
    primary_arrivals = np.asarray(primary_arrivals, dtype=float)
    committed_secondary = np.asarray(committed_secondary, dtype=float)
    secondary_caps = np.asarray(secondary_caps, dtype=float)
    if bom.shape != (materials, products) or primary_arrivals.shape != (n, materials):
        raise ValueError('APP dimensions mismatch')
    arrays = [demand, capacity, bom, opening_raw, opening_finished, opening_backlog,
              primary_arrivals, committed_secondary, secondary_caps]
    if not all(np.isfinite(a).all() and np.min(a) >= 0 for a in map(np.asarray, arrays)):
        raise ValueError('APP inputs must be finite nonnegative quantities')
    if lead < 1 or not np.allclose(committed_secondary[0], 0):
        raise ValueError('Positive lead required; current secondary receipts already in opening_raw')
    if costs['raw_holding_per_die_period'] != 0 or costs['terminal_inventory_salvage'] != 0:
        raise ValueError('This model implements the approved zero raw-holding/salvage policy')
    names, indices = [], {}
    for family, width in [('p', products), ('i', products), ('b', products), ('q', materials)]:
        for k in range(n):
            for j in range(width):
                indices[family,k,j] = len(names)
                names.append(f'{family}_{k+1}_{j+1}')
    size = len(names)
    obj, upper = np.zeros(size), np.full(size, np.inf)
    for k in range(n):
        for j in range(products):
            obj[indices['p',k,j]] = costs['production_per_product_unit']
            obj[indices['i',k,j]] = costs['finished_holding_per_product_unit_period']
            obj[indices['b',k,j]] = costs['backlog_per_product_unit_period'] + (terminal_penalty if k == n-1 else 0)
        for i in range(materials):
            obj[indices['q',k,i]] = costs['secondary_purchase_per_die']
            upper[indices['q',k,i]] = secondary_caps[k,i] if k+lead < n else 0
    eq, rhs, ub, limits = [], [], [], []
    for k in range(n):
        for j in range(products):
            row = np.zeros(size)
            row[indices['i',k,j]], row[indices['b',k,j]], row[indices['p',k,j]] = 1,-1,-1
            if k:
                row[indices['i',k-1,j]], row[indices['b',k-1,j]] = -1,1
            eq.append(row)
            rhs.append(-demand[k,j] + (opening_finished[j]-opening_backlog[j] if k == 0 else 0))
        row = np.zeros(size)
        for j in range(products): row[indices['p',k,j]] = 1
        ub.append(row); limits.append(capacity[k])
        for i in range(materials):
            row = np.zeros(size)
            for u in range(k+1):
                for j in range(products): row[indices['p',u,j]] = bom[i,j]
            # q[u] is usable at relative period u+lead, never earlier.
            for u in range(max(0,k-lead+1)): row[indices['q',u,i]] = -1
            ub.append(row)
            limits.append(opening_raw[i] + primary_arrivals[:k,i].sum() + committed_secondary[:k+1,i].sum())
    return AppLP(names, obj, np.asarray(eq), np.asarray(rhs), np.asarray(ub), np.asarray(limits), upper, indices, (n,products,materials))


def expression(values, names):
    terms = []
    for coefficient, name in zip(values, names):
        if coefficient:
            terms.append(f'{"+" if coefficient > 0 else "-"} {abs(coefficient):.17g} {name}')
    return ' '.join(terms) if terms else f'0 {names[0]}'


def export_lp(model, path):
    lines = ['Minimize', ' total_cost: '+expression(model.objective, model.names), 'Subject To']
    lines += [f' balance_{i}: {expression(row,model.names)} = {rhs:.17g}'
              for i,(row,rhs) in enumerate(zip(model.a_eq,model.b_eq),1)]
    lines += [f' limit_{i}: {expression(row,model.names)} <= {rhs:.17g}'
              for i,(row,rhs) in enumerate(zip(model.a_ub,model.b_ub),1)]
    lines.append('Bounds')
    for name, maximum in zip(model.names, model.upper):
        lines.append(f' 0 <= {name} <= {maximum:.17g}' if np.isfinite(maximum) else f' {name} >= 0')
    lines += ['End','']
    Path(path).write_text('\n'.join(lines), encoding='ascii')


def residuals(model, solution):
    finite = np.isfinite(model.upper)
    return {'max_equality_residual':float(np.max(np.abs(model.a_eq@solution-model.b_eq))),
            'max_inequality_violation':float(max(0,np.max(model.a_ub@solution-model.b_ub))),
            'max_nonnegativity_violation':float(max(0,-np.min(solution))),
            'max_upper_bound_violation':float(max(0,np.max(solution[finite]-model.upper[finite]))) if finite.any() else 0.}


def solve_cplex(model, directory, executable, threads=1, time_limit=120, tolerance=1e-5):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    export_lp(model, directory/'model.lp')
    start = time.perf_counter()
    command = [str(executable), '-c', 'read model.lp', f'set threads {threads}',
               f'set timelimit {time_limit}', 'set lpmethod 2', 'optimize', 'write solution.sol', 'quit']
    result = subprocess.run(command, cwd=directory, capture_output=True, timeout=time_limit+30,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    log = (result.stdout+result.stderr).decode('utf-8', errors='replace')
    (directory/'solver_console.log').write_text(log, encoding='utf-8')
    if result.returncode != 0 or not (directory/'solution.sol').is_file():
        raise RuntimeError(f'CPLEX failed in {directory}; see solver_console.log')
    root = ET.parse(directory/'solution.sol').getroot()
    header = root.find('header').attrib
    if header.get('solutionStatusValue') != '1' or header.get('primalFeasible') != '1':
        raise RuntimeError(f'CPLEX not optimal/feasible: {header}')
    variables = {v.attrib['name']:float(v.attrib['value']) for v in root.find('variables')}
    if set(variables) != set(model.names):
        raise ValueError('CPLEX solution variables differ from the exported LP')
    solution = np.asarray([variables[name] for name in model.names])
    if not np.isfinite(solution).all(): raise ValueError('Nonfinite solution')
    audit = residuals(model, solution)
    if max(audit.values()) > tolerance: raise ValueError(f'CPLEX residual exceeds tolerance: {audit}')
    objective = float(model.objective @ solution)
    if abs(objective-float(header['objectiveValue'])) > tolerance:
        raise ValueError('CPLEX objective does not match independently recomputed objective')
    # Never replace the original solution before checking all residuals.
    clean = np.maximum(solution,0)
    n, products, materials = model.shape
    plan = {f:np.asarray([[clean[model.indices[f,k,j]] for j in range(width)] for k in range(n)])
            for f,width in [('p',products),('i',products),('b',products),('q',materials)]}
    audit.update(status='optimal', objective=objective, seconds=time.perf_counter()-start,
                 solver_version='12.8.0.0', algorithm='dual_simplex', threads=threads)
    return plan, audit
