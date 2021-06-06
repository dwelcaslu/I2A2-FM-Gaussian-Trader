import os, sys
sys.path.append('../src/')

import copy
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import stockmarket as smkt


class TraderArena:
    def __init__(self, target_names, n_gens=10, n_agents=10, mut_perc=0.1, patience=100) -> None:
        # Input parameters:
        self.target_names = target_names
        self.n_gens = n_gens
        self.n_agents = n_agents
        self.mut_perc = mut_perc
        self.patience = patience
        # Outputs:
        self.generations = []
        self.wealth = []
        self.y_template_opt = None
        self.est_opt = None
    
    def run(self, df_train: pd.DataFrame, params: dict):
        templates = []
        patience_count = 0
        for i in range(self.n_gens):
            if not templates:
                templates = self.populate_templates(df_train.shape[0])
            else:
                templates = self.mutate_templates(templates)
            # Re-training the model with different y_train values:
            X_train = df_train[params['features_names']]
            wealth_array = []
            estimators = []
            for y_train in templates:
                est = copy.deepcopy(params['estimator'])
                est.fit(X_train, y_train)
                mkt_op = smkt.MarkerOperator(est, params['features_names'],
                                             initial_cash=params.get('initial_cash', 1000),
                                             initial_stocks= params.get('initial_stocks', 0),
                                             daily_negotiable_perc=params.get('daily_negotiable_perc', 0.2),
                                             min_stocks_op=params.get('min_stocks_op', 1),
                                             broker_taxes=params.get('broker_taxes', 0))
                op_results = mkt_op.run(df_train)
                wealth_array.append(op_results['wealth'].values[-1])
                estimators.append(est)
            
            templates = [templates[np.argmax(wealth_array)]]
            estimators = [estimators[np.argmax(wealth_array)]]
            self.generations.append(i+1)
            self.wealth.append(max(wealth_array))
            if i == 0:
                print('Initial wealth:', op_results['wealth'].values[0])
            print('Generation number:', i+1, 'wealth:', round(max(wealth_array), 2))
            if i > 0 and self.wealth[-1] == self.wealth[-2]:
                patience_count += 1
            else:
                patience_count = 0
            if patience_count == self.patience:
                break
        self.y_template_opt = templates[0]
        self.est_opt = estimators[0]
        return self.est_opt
    
    def populate_templates(self, templates_len):
        templates = [np.random.choice([0, 1, 2], templates_len) for _ in range(self.n_agents)]
        return templates       

    def mutate_templates(self, templates):
        mutated_templates = []
        for temp in templates:
            n_mutations = int(self.mut_perc * len(temp))
            mutated_templates.append(copy.deepcopy(temp))
            for _ in range(self.n_agents):
                mut_idxs = np.random.choice([i for i in range(len(temp))], n_mutations)
                mut_values = np.random.choice(self.target_names, n_mutations)
                mutated_templates.append(copy.deepcopy(temp))
                mutated_templates[-1][mut_idxs] = mut_values
        return mutated_templates
    
    def plot_evolution(self, figsize=(12, 8)):
        plt.figure(figsize=figsize)
        plt.plot(self.generations, self.wealth, zorder=2)
        plt.xlim([min(self.generations), max(self.generations)])
        plt.ylabel('wealth evolution')
        plt.grid()
        plt.tight_layout()
        plt.show()
