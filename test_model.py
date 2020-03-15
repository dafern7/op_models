# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:55:58 2019

@author: IST_1
"""
import numpy as np
from pyomo.environ import *
import pandas as pd
import datetime



model = ConcreteModel()
model.x = Var(domain=NonNegativeIntegers, initialize=0)
model.y = Var(domain=NonNegativeIntegers, initialize=0)

def power_constraint(model):
    return 4*model.x+3.5*model.y==15
model.power_constraint = Constraint(rule=power_constraint)

    
expenses = model.x*5065843+model.y*4637561
model.expenses = expenses
model.objective = Objective(expr=expenses, sense=minimize)

    # Solve the model
solver = SolverFactory('gurobi')
solver.solve(model)