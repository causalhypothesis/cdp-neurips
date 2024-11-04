from copy import deepcopy
from itertools import cycle
from typing import Any, Dict, List, Optional, AnyStr, Tuple

import numpy as np
import pandas as pd
from dowhy import gcm
from sklearn import base
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from scmtools.utils import copy_causal_model_with_frozen_mechanisms
from scmtools.augment import augment_causal_model_with_black_box, sample_augmented_counterfactuals


class CausalDependencePlotter:
    def __init__(
        self,
        black_box_model: base.BaseEstimator, 
        fit_causal_model: gcm.InvertibleStructuralCausalModel,
        explanatory_X_data: pd.DataFrame,
        outcome_name: AnyStr,
        prefit_black_box: Optional[bool]=False,
        black_box_X_train: Optional[pd.DataFrame]=None, 
        black_box_y_train: Optional[np.array]=None,
        individual_curve_color: Optional[AnyStr]='gray',
        average_curve_color: Optional[AnyStr]='blue',
        individual_alpha_level: Optional[float]=0.35,
        average_alpha_level: Optional[float]=1.0,
        linestyle: Optional[AnyStr]='solid'
    ) -> None:
        """Class to support plotting total, partially controlled, natural direct, and natural indirect effect functions.
        
        NOTE: outcome_name cannot be a source node
        
        Args:
            black_box_model (base.BaseEstimator): an sklearn estimator that predicts the outcome variable
            fit_causal_model (gcm.InvertibleStructuralCausalModel): an already fit causal model for every variable in explanatory_X_data
            explanatory_X_data (pd.DataFrame): explanatory data whose columns are a superset of the columns of black_box_X_train
            outcome_name (AnyStr): name of the outcome variable (used for plotting)
            prefit_black_box (bool, optional): whether or not black_box_model is already fit. Defaults to False.
            black_box_X_train (pd.DataFrame, optional): data the black box is trained on, excluding the outcome variable. Must be provided if prefit_black_box=False. Defaults to None.
            black_box_y_train (np.array, optional): outcome varaible data the black box is trained on. Must be provided if prefit_black_box=False. Defaults to None.
            individual_curve_color (AnyStr, optional): matplotlib.pyplot color to use for individual counterfactual curves
            average_curve_color (AnyStr, optional): matplotlib.pyplot color to use for average counterfactual curves
            individual_alpha_level (float, optional): matplotlib.pyplot alpha level to use for individual counterfactual curves
            average_alpha_level (float, optional): matplotlib.pyplot alpha level to use for average counterfactual curves
            linestyle (AnyStr, optional): matplotlib.pyplot linestyle to use for all curves

        Raises:
            ValueError: black_box_X_train and black_box_y_train must be provided if prefit_black_box is False
            Warning: black_box_X_train and black_box_y_train should not be provided if prefit_black_box is True
        """
        self.individual_curve_color = individual_curve_color
        self.average_curve_color = average_curve_color
        self.individual_alpha_level = individual_alpha_level
        self.average_alpha_level = average_alpha_level
        self.linestyle = linestyle
        
        if prefit_black_box:
            if black_box_X_train is not None or black_box_y_train is not None:
                raise Warning('black_box_X_train and black_box_y_train should not be provided if prefit_black_box is True')
            
            # NOTE: assumes black_box_model is an sklearn.base.BaseEstimator
            self.black_box_feature_names = getattr(black_box_model, 'feature_names_in_', None)  
            self.fit_black_box_model = black_box_model
        else:
            if black_box_X_train is None or black_box_y_train is None:
                raise ValueError('black_box_X_train and black_box_y_train must be provided if prefit_black_box is False')
            self.black_box_feature_names = black_box_X_train.columns.tolist()
            black_box_model.fit(X=black_box_X_train, y=black_box_y_train)
            self.fit_black_box_model = black_box_model
        
        self.outcome_name = outcome_name
        
        explanatory_data = explanatory_X_data.copy()
        explanatory_data[self.outcome_name] = self.fit_black_box_model.predict(explanatory_X_data[self.black_box_feature_names])
        self.explanatory_data = explanatory_data
        
        # NOTE: fit_causal_model is modified by the function rather than returned
        self.black_box_augmented_causal_model = augment_causal_model_with_black_box(
            fit_causal_model=fit_causal_model, 
            outcome_name=self.outcome_name, 
            fit_black_box_model=self.fit_black_box_model, 
            black_box_feature_names=self.black_box_feature_names
        )

    
    def get_treatment_vals(
        self, 
        treatment_var: AnyStr
    ) -> None:
        """Get possible values for treatment_var based on the observed range in the explanatory data.

        Args:
            treatment_var (AnyStr): name of the treatment variable column in explanatory_X_data

        Returns:
            np.array: possible values for treatment_var
        """
        treatment_vals = np.linspace(
            self.explanatory_data[treatment_var].min(), 
            self.explanatory_data[treatment_var].max(), 
            100
        )
        return treatment_vals
    

    def _compute_total_effect_function(self, treatment_var: AnyStr) -> pd.DataFrame:
        """Helper function to compute total effect function. See plot_total_effect() docstring."""
        treatment_vals = self.get_treatment_vals(treatment_var=treatment_var)
        
        cf_outcomes_dict = {}
        for treatment_val in tqdm(treatment_vals, desc='Sampling TE counterfactuals'):
            intervention_dict = {treatment_var: lambda x: treatment_val}
            cf_data = sample_augmented_counterfactuals(
                outcome_name=self.outcome_name, 
                black_box_augmented_causal_model=self.black_box_augmented_causal_model,
                intervention_dict=intervention_dict,
                observed_data=self.explanatory_data
            )

            outcome_vals = cf_data[self.outcome_name].copy()
            cf_outcomes_dict[treatment_val] = outcome_vals

        cf_outcomes_df = pd.DataFrame(cf_outcomes_dict)
        return cf_outcomes_df
    

    def plot_total_effect(
        self,
        treatment_var: AnyStr,
        axis: Optional[plt.axis]=None
    ) -> None:
        """Plot the total effect function with interventions on treatment_var.

        Args:
            treatment_var (AnyStr): the treatment variable
            axis (plt.axis, optional): an existing matplotlib.pyplot.axis to add the plot to. Defaults to None.
        """
        if axis is None:
            plt.figure()
            axis = plt.gca()
        
        cf_outcomes_df = self._compute_total_effect_function(treatment_var=treatment_var)

        for i in tqdm(list(range(len(cf_outcomes_df))), desc='Plotting total effect'):
            cf_outcomes_df.loc[i, :].plot(color=self.individual_curve_color, ax=axis, alpha=self.individual_alpha_level, linewidth=0.8, linestyle=self.linestyle)
        cf_outcomes_df.mean(axis=0).plot(color=self.average_curve_color, ax=axis, alpha=self.average_alpha_level, linewidth=2, linestyle=self.linestyle)
        axis.set_xlabel(treatment_var)
        axis.set_ylabel(self.outcome_name)
        axis.set_title('Total Dependence')
        
    
    def _compute_partially_controlled_effect_function(
            self,
            treatment_var: AnyStr, 
            control_vars: Tuple[AnyStr], 
            control_tuples: List[Tuple[Any]]
        ) -> Dict[AnyStr, Dict[AnyStr, List[Any]]]:
        """Helper function to compute conrolled effect function. See plot_controlled_effect() docstring."""
        treatment_vals = self.get_treatment_vals(treatment_var=treatment_var)
        
        cf_outcomes_dict_of_dicts = {}
        for control_tuple in tqdm(control_tuples, desc='Sampling PCE counterfactuals'):
            cf_outcomes_dict = {}
            for treatment_val in treatment_vals:
                intervention_dict = {treatment_var: lambda x: treatment_val}
                for i, control_var in enumerate(control_vars):
                    intervention_dict[control_var] = lambda x: control_tuple[i]
                    
                cf_data = sample_augmented_counterfactuals(
                    outcome_name=self.outcome_name, 
                    black_box_augmented_causal_model=self.black_box_augmented_causal_model,
                    intervention_dict=intervention_dict,
                    observed_data=self.explanatory_data
                )

                outcome_vals = cf_data[self.outcome_name].copy()
                cf_outcomes_dict[treatment_val] = outcome_vals

            cf_outcomes_dict_of_dicts[control_tuple] = cf_outcomes_dict
            
        return cf_outcomes_dict_of_dicts
    
    
    def plot_controlled_effect(
        self,
        treatment_var: AnyStr, 
        control_vars: Tuple[AnyStr], 
        control_tuples: List[Tuple[Any]],
        combine_plots: bool,
        axis: plt.axis=None
        
    ) -> None:
        """Plot the controlled effect function given interventions on treatment_var.
        The variables in control_vars will be set simultaneously to the values in control_tuples.
        Consider the following example with 3 control variables across four settings:
        # control_vars = (control_var_1, control_var_2, control_var_3)
        # control_tuples = [
        #     (val_1_1, val_2_1, val_3_1),
        #     (val_1_2, val_2_2, val_3_2),
        #     (val_1_3, val_2_3, val_3_3),
        #     (val_1_4, val_2_4, val_3_4)
        # ]
        For the above example, the first plot (or line, depending on combine_plots) will vary treatment_var 
        using do(treatment_var, control_var_1=val_1_1, control_var_2=val_2_1, control_var_3=vaL_3_1), the
        second plot will use do(treatment_var, control_var_1=val_1_2, control_var_2=val_2_2, control_var_3=vaL_3_2),
        and so on. If combine_plots=True, one value -- the mean of the outcome across all observations -- is plotted, 
        rather than one line for each observation, and all means are shown on the same plot. If combine_plots=False,
        a separate plot is made for each tuple in control_tuples.

        Args:
            treatment_var (AnyStr): the treatment variable
            control_vars (Tuple[AnyStr]): the variables to control
            control_tuples (List[Tuple[Any]]): a list of tuples specifying how to set each variable in control_vars
            combine_plots (bool): whether to combine the plots into one, showing only means, or instead make a separate
                                  plot for each tuple in control_tuples
            axis (plt.axis, optional): an existing matplotlib.pyplot.axis to add the plot to. Defaults to None.
        """
        cf_outcomes_dict_of_dicts = self._compute_partially_controlled_effect_function(
            treatment_var=treatment_var,
            control_vars=control_vars,
            control_tuples=control_tuples
        )
        
        color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        linestyle_cycle = cycle(['-', '--', ':', '-.'])

        for control_tuple in control_tuples:
            cf_outcomes_dict = cf_outcomes_dict_of_dicts[control_tuple]
            cf_outcomes_df = pd.DataFrame(cf_outcomes_dict)
            if not combine_plots:
                plt.figure()
                for i in tqdm(list(range(len(cf_outcomes_df))), 
                            desc='Plotting controlled effect'):
                    cf_outcomes_df.loc[i, :].plot(color=self.individual_curve_color, alpha=self.individual_alpha_level, linewidth=0.8, linestyle=self.linestyle)
                cf_outcomes_df.mean(axis=0).plot(color=self.average_curve_color, alpha=self.average_alpha_level, linewidth=2, linestyle=self.linestyle)
                plot_title = ', '.join(f'{control_var}={value}' for control_var, value in zip(control_vars, control_tuple))
                plt.title(plot_title)
                plt.xlabel(treatment_var)
                plt.ylabel(self.outcome_name)
            else:
                if axis is None:
                    plt.figure()
                    axis = plt.gca()
                if len(control_tuple) == 1:
                    label = control_tuple[0]
                else:
                    label = control_tuple
                color = next(color_cycle)
                linestyle = next(linestyle_cycle)
                cf_outcomes_df.mean(axis=0).plot(label=label, ax=axis, color=color, linestyle=linestyle) 
        if combine_plots:
            axis.set_xlabel(treatment_var)
            axis.set_ylabel(self.outcome_name)
            axis.set_title('Partially Controlled Dependence')
            if len(control_vars) == 1:
                legend_title = control_vars[0]
            else:
                legend_title = str(control_vars).replace("'", '')
            axis.legend(title=legend_title)
            
    
    def _compute_natural_direct_effect_function(self, treatment_var: AnyStr) -> pd.DataFrame:
        """Helper function to compute natural direct effect function. See plot_direct_effect() docstring."""  
        treatment_vals = self.get_treatment_vals(treatment_var=treatment_var)
        
        children_of_treatment = list(
            self.black_box_augmented_causal_model.graph.successors(treatment_var)
        )
        if self.outcome_name in children_of_treatment:
            children_of_treatment.remove(self.outcome_name)
            
        intervention_dict = {}
        for child_var in children_of_treatment:
            intervention_dict[child_var] = lambda x: x
            
        frozen_causal_model = copy_causal_model_with_frozen_mechanisms(
            causal_model=self.black_box_augmented_causal_model,
            freeze_mechanisms_of=children_of_treatment
        )
        
        cf_outcomes_dict = {}
        for treatment_val in tqdm(treatment_vals, desc='Sampling NDE counterfactuals'):  
            intervention_dict[treatment_var] = lambda x: treatment_val
                              
            cf_data = sample_augmented_counterfactuals(
                outcome_name=self.outcome_name, 
                black_box_augmented_causal_model=frozen_causal_model,
                intervention_dict=intervention_dict,
                observed_data=self.explanatory_data
            )

            outcome_vals = cf_data[self.outcome_name].copy()
            cf_outcomes_dict[treatment_val] = outcome_vals

        cf_outcomes_df = pd.DataFrame(cf_outcomes_dict)
        
        return cf_outcomes_df
    
    
    def plot_direct_effect(
        self,
        treatment_var: AnyStr,
        axis: plt.axis=None
    ) -> None:
        """Plot the natural direct effect function with interventions on treatment_var.

        Args:
            treatment_var (AnyStr): the treatment variable
            axis (plt.axis, optional): an existing matplotlib.pyplot.axis to add the plot to. Defaults to None.
        """
        if axis is None:
            plt.figure()
            axis = plt.gca()
        
        cf_outcomes_df = self._compute_natural_direct_effect_function(treatment_var=treatment_var)

        for i in tqdm(list(range(len(cf_outcomes_df))), desc='Plotting direct effect'):
            cf_outcomes_df.loc[i, :].plot(color=self.individual_curve_color, ax=axis, alpha=self.individual_alpha_level, linewidth=0.8, linestyle=self.linestyle)
        cf_outcomes_df.mean(axis=0).plot(color=self.average_curve_color, ax=axis, alpha=self.average_alpha_level, linewidth=2, linestyle=self.linestyle)
        axis.set_xlabel(treatment_var)
        axis.set_ylabel(self.outcome_name)
        axis.set_title('Natural Direct Dependence')
        
    
    def _compute_natural_indirect_effect_function(self, treatment_var: AnyStr) -> pd.DataFrame:
        """Helper function to compute the natural indirect effect function. See plot_indirect_effect() docstring."""
        treatment_vals = self.get_treatment_vals(treatment_var=treatment_var)
        
        children_of_treatment = list(
            self.black_box_augmented_causal_model.graph.successors(treatment_var)
        )
        if self.outcome_name in children_of_treatment:
            children_of_treatment.remove(self.outcome_name)
            
        intervention_dict = {}
        for child_var in children_of_treatment:
            intervention_dict[child_var] = lambda x: x
            
        frozen_causal_model = copy_causal_model_with_frozen_mechanisms(
            causal_model=self.black_box_augmented_causal_model,
            freeze_mechanisms_of=children_of_treatment
        )
        
        intervention_dict[treatment_var] = lambda x: x
        
        cf_outcomes_dict = {}
        for treatment_val in tqdm(treatment_vals, desc='Sampling NIE counterfactuals'):  
            children_intervention_dict = {treatment_var: lambda x: treatment_val}
            cf_children_data = sample_augmented_counterfactuals(
                outcome_name=self.outcome_name, 
                black_box_augmented_causal_model=self.black_box_augmented_causal_model,
                intervention_dict=children_intervention_dict,
                observed_data=self.explanatory_data
            )
            
            observed_data = self.explanatory_data.copy()
            for child_var in children_of_treatment:
                observed_data[child_var] = cf_children_data[child_var].values.copy()       
                              
            cf_data = sample_augmented_counterfactuals(
                outcome_name=self.outcome_name, 
                black_box_augmented_causal_model=frozen_causal_model,
                intervention_dict=intervention_dict,
                observed_data=observed_data
            )

            outcome_vals = cf_data[self.outcome_name].copy()
            cf_outcomes_dict[treatment_val] = outcome_vals

        cf_outcomes_df = pd.DataFrame(cf_outcomes_dict)
        
        return cf_outcomes_df
    
    
    def plot_indirect_effect(
        self,
        treatment_var: AnyStr,
        axis: plt.axis=None
    ) -> None:
        """Plot the natural indirect effect function with interventions on treatment_var.

        Args:
            treatment_var (AnyStr): the treatment variable
            axis (plt.axis, optional): an existing matplotlib.pyplot.axis to add the plot to. Defaults to None.
        """
        if axis is None:
            plt.figure()
            axis = plt.gca()
        
        cf_outcomes_df = self._compute_natural_indirect_effect_function(treatment_var=treatment_var)

        for i in tqdm(list(range(len(cf_outcomes_df))), desc='Plotting indirect effect'):
            cf_outcomes_df.loc[i, :].plot(color=self.individual_curve_color, ax=axis, alpha=self.individual_alpha_level, linewidth=0.8, linestyle=self.linestyle)
        cf_outcomes_df.mean(axis=0).plot(color=self.average_curve_color, ax=axis, alpha=self.average_alpha_level, linewidth=2, linestyle=self.linestyle)
        axis.set_xlabel(treatment_var)
        axis.set_ylabel(self.outcome_name)
        axis.set_title('Natural Indirect Dependence')
        

class UncertainCausalDependencePlotter:
    def __init__(
        self,
        black_box_model: base.BaseEstimator, 
        causal_model_list: List[gcm.InvertibleStructuralCausalModel],
        explanatory_X_data: pd.DataFrame,
        outcome_name: AnyStr,
        prefit_black_box: Optional[bool]=False,
        black_box_X_train: Optional[pd.DataFrame]=None, 
        black_box_y_train: Optional[np.array]=None
    ) -> None:
        """Class to support plotting total, natural direct, and natural indirect effect functions with a list of 
        candidate causal models instead of one causal model.
        
        Args:
            black_box_model (base.BaseEstimator): an sklearn estimator that predicts the outcome variable
            causal_model_list (List[gcm.InvertibleStructuralCausalModel]): a list of already fit causal models for every variable in explanatory_X_data
            explanatory_X_data (pd.DataFrame): explanatory data whose columns are a superset of the columns of black_box_X_train
            outcome_name (AnyStr): name of the outcome variable (used for plotting)
            prefit_black_box (bool, optional): whether or not black_box_model is already fit. Defaults to False.
            black_box_X_train (pd.DataFrame, optional): data the black box is trained on, excluding the outcome variable. Must be provided if prefit_black_box=False. Defaults to None.
            black_box_y_train (np.array, optional): outcome varaible data the black box is trained on. Must be provided if prefit_black_box=False. Defaults to None.
        
        Raises:
            ValueError: black_box_X_train and black_box_y_train must be provided if prefit_black_box is False
            Warning: black_box_X_train and black_box_y_train should not be provided if prefit_black_box is True
        """
        self.causal_dependence_plotter_list = [
            CausalDependencePlotter(
                black_box_model=black_box_model, 
                fit_causal_model=fit_causal_model,
                explanatory_X_data=explanatory_X_data,
                outcome_name=outcome_name,
                prefit_black_box=prefit_black_box,
                black_box_X_train=black_box_X_train, 
                black_box_y_train=black_box_y_train
            )
            for fit_causal_model in causal_model_list
        ]
        
        self.outcome_name = outcome_name
        
    def plot_effect_with_envelope(
        self,
        treatment_var: AnyStr,
        type_of_effect: AnyStr,
        axis: Optional[plt.axis]=None
    ) -> None:
        """Helper function for total, natural direct, and natural indirect effect plotting.
        Argument type_of_effect can be one of ['total', 'direct', 'indirect'].
        """
        if axis is None:
            plt.figure()
            axis = plt.gca()
        
        outcome_df_list = []
        for i, plotter in enumerate(self.causal_dependence_plotter_list):
            if type_of_effect == 'total':
                effect_function = plotter._compute_total_effect_function
            elif type_of_effect == 'direct':
                effect_function = plotter._compute_natural_direct_effect_function
            elif type_of_effect == 'indirect':
                effect_function = plotter._compute_natural_indirect_effect_function
                
            cf_outcomes_df = effect_function(treatment_var=treatment_var)
            cf_outcomes_df['causal_model_num'] = i
            outcome_df_list.append(cf_outcomes_df)
        
        all_outcomes_df = pd.concat(outcome_df_list)
            
        mean_outcomes = all_outcomes_df.groupby(['causal_model_num']).mean()
        
        x = [float(val) for val in mean_outcomes.columns]
        y_min = mean_outcomes.min(axis=0).values
        y_max = mean_outcomes.max(axis=0).values
        axis.fill_between(x, y1=y_min, y2=y_max, alpha=0.5, color='gray')
        axis.set_xlabel(treatment_var)
        axis.set_ylabel(self.outcome_name)

        color = 'black'
        linestyle_cycle = cycle(['-', '--', ':', '-.'])
        for i in range(len(self.causal_dependence_plotter_list)):
            linestyle = next(linestyle_cycle)
            y_vals = mean_outcomes.iloc[i, :].values
            axis.plot(x, y_vals, alpha=0.5, label=f'Model {i + 1}', color=color, linestyle=linestyle)
        axis.legend()
        
        if type_of_effect == 'total':
            plot_title = 'Total Dependence'
        elif type_of_effect == 'direct':
            plot_title = 'Natural Direct Dependence'
        elif type_of_effect == 'indirect':
            plot_title = 'Natural Indirect Dependence'
        axis.set_title(plot_title)
    
    def plot_total_effect(
        self,
        treatment_var: AnyStr,
        axis: Optional[plt.axis]=None
    ) -> None:
        """Plot the total effect function with interventions on treatment_var.

        Args:
            treatment_var (AnyStr): the treatment variable
            axis (plt.axis, optional): an existing matplotlib.pyplot.axis to add the plot to. Defaults to None.
        """
        self.plot_effect_with_envelope(treatment_var=treatment_var, type_of_effect='total', axis=axis)
        
    def plot_direct_effect(
        self,
        treatment_var: AnyStr,
        axis: Optional[plt.axis]=None
    ) -> None:
        """Plot the natural direct effect function with interventions on treatment_var.

        Args:
            treatment_var (AnyStr): the treatment variable
            axis (plt.axis, optional): an existing matplotlib.pyplot.axis to add the plot to. Defaults to None.
        """
        self.plot_effect_with_envelope(treatment_var=treatment_var, type_of_effect='direct', axis=axis)
        
    def plot_indirect_effect(
        self,
        treatment_var: AnyStr,
        axis: Optional[plt.axis]=None
    ) -> None:
        """Plot the natural indirect effect function with interventions on treatment_var.

        Args:
            treatment_var (AnyStr): the treatment variable
            axis (plt.axis, optional): an existing matplotlib.pyplot.axis to add the plot to. Defaults to None.
        """
        self.plot_effect_with_envelope(treatment_var=treatment_var, type_of_effect='indirect', axis=axis)
