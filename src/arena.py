import os, sys
sys.path.append('../src/')

import copy
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE 

import stockmarket as smkt


plt.rcParams.update({'font.size': 12})


class TraderArena:
    def __init__(self, target_names: list, target_prop: list = None,
                 n_gens: int = 10, init_population: int = 100, n_mutations: int = 10,
                 mut_perc: float = 0.1, patience: int = 100, train_size: float = 0.6) -> None:
        # Input parameters:
        self.target_names = target_names
        self.target_prop = target_prop if target_prop is not None else np.array([1/len(target_names) for _ in target_names])
        self.n_gens = n_gens
        self.init_population = init_population
        self.n_mutations = n_mutations
        self.mut_perc = mut_perc
        self.patience = patience
        self.train_size = train_size
        # Auxiliar variables:
        self.generations = []
        self.wealth_train = []
        self.wealth_valid = []
        self.initial_cash = None
        # Outputs:
        self.y_template_opt = None
        self.est_opt = None
    
    def run(self, df_train: pd.DataFrame, params: dict, rebalance: bool = False):
        templates = []
        wealth_train_valid = []
        patience_count = 0
        for i in range(self.n_gens):
            if not templates:
                templates = self.populate_templates(df_train.shape[0])
            else:
                templates = self.mutate_templates(templates, wealth_train_valid)
            # Re-training the model with different y_true values
            idx_split = int(self.train_size * df_train.shape[0])
            x_train = df_train[params['features_names']][0:idx_split]
            wealth_array_train = []
            wealth_array_valid = []
            estimators = []
            for y_true in templates:
                y_train = y_true[0:idx_split]
                # Re-balancing the classes and training the model
                if rebalance:
                    sm = SMOTE(random_state=42)
                    x_train, y_train = sm.fit_resample(x_train, y_train)
                est = copy.deepcopy(params['estimator'])
                est.fit(x_train, y_train)
                mkt_op = smkt.MarkerOperator(est, params['features_names'],
                                             initial_cash=params.get('initial_cash', 1000),
                                             initial_stocks= params.get('initial_stocks', 0),
                                             daily_negotiable_perc=params.get('daily_negotiable_perc', 0.5),
                                             min_stocks_op=params.get('min_stocks_op', 1),
                                             broker_taxes=params.get('broker_taxes', 0))
                # Executing the model in the train and validation data
                op_results_train = mkt_op.run(df_train[0:idx_split])
                op_results_valid = mkt_op.run(df_train[idx_split:])
                wealth_array_train.append(op_results_train['wealth'].values[-1])
                wealth_array_valid.append(op_results_valid['wealth'].values[-1])
                estimators.append(est)
                if len(wealth_train_valid) == 0 or op_results_valid['wealth'].values[-1] == max(wealth_array_valid):
                    wealth_train_valid = np.concatenate((op_results_train['wealth'].values[0:-1], op_results_valid['wealth'].values[0:-1]))
            # Selecting the generation best results
            templates = [templates[np.argmax(wealth_array_valid)]]
            estimators = [estimators[np.argmax(wealth_array_valid)]]
            self.generations.append(i+1)
            self.wealth_train.append(max(wealth_array_train))
            self.wealth_valid.append(max(wealth_array_valid))
            if i == 0:
                print('Initial wealth:', op_results_train['wealth'].values[0])
                self.initial_cash = op_results_train['wealth'].values[0]
            print(f'Generation {i+1}: train wealth {round(max(wealth_array_train), 2)} valid wealth {round(max(wealth_array_valid), 2)}')
            if i > 0 and self.wealth_valid[-1] == self.wealth_valid[-2]:
                patience_count += 1
            else:
                patience_count = 0
            if patience_count == self.patience:
                break
        self.y_template_opt = templates[0]
        self.est_opt = estimators[0]
        return self.est_opt
    
    def populate_templates(self, templates_len):
        templates = [np.random.choice(self.target_names, templates_len, p=self.target_prop, replace=True) for _ in range(self.init_population)]
        return templates

    def mutate_templates(self, templates, wealth_train_valid):
        idx_grad_neg = np.where(np.gradient(wealth_train_valid) <= 0)[0]
        if idx_grad_neg.shape[0] == wealth_train_valid.shape[0]:
            idx_grad_neg = np.array([])
        mutated_templates = []
        for temp in templates:
            n_mutations = int(self.mut_perc * len(temp))
            mutated_templates.append(copy.deepcopy(temp))
            for _ in range(int(self.n_mutations/2)):
                try:
                    mut_idxs = np.random.choice(idx_grad_neg, n_mutations, replace=False if idx_grad_neg.shape[0] > n_mutations else True)
                except ValueError:
                    mut_idxs = np.random.choice([i for i in range(len(temp))], n_mutations, replace=False)
                mut_values = np.random.choice(self.target_names, n_mutations, p=self.target_prop, replace=True)
                mutated_templates.append(copy.deepcopy(temp))
                mutated_templates[-1][mut_idxs] = mut_values
            for _ in range(int(self.n_mutations/2)):
                mut_idxs = np.random.choice([i for i in range(len(temp))], n_mutations, replace=False)
                mut_values = np.random.choice(self.target_names, n_mutations, p=self.target_prop, replace=True)
                mutated_templates.append(copy.deepcopy(temp))
                mutated_templates[-1][mut_idxs] = mut_values
        return mutated_templates

    def plot_evolution(self, figsize=(12, 8), normalize=True):
        div = self.initial_cash if normalize else 1
        plt.figure(figsize=figsize)
        plt.title(f'{"Normalized " if normalize else ""}Wealth ratio evolution')
        plt.plot(self.generations, self.wealth_train/div, zorder=2, label='train')
        plt.plot(self.generations, self.wealth_valid/div, zorder=2, label='validation')
        plt.xlim([min(self.generations), max(self.generations)])
        plt.xlabel('generation number')
        plt.ylabel('wealth evolution')
        plt.legend(loc='best', prop={'size': 10})
        plt.grid()
        plt.tight_layout()
        plt.show()
