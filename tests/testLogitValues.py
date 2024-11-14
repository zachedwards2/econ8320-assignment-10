import unittest
import json
import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Logit
import statsmodels.api as sm

with open("Lesson.ipynb", "r") as file:
    f_str = file.read()

doc = json.loads(f_str)

code = [i for i in doc['cells'] if i['cell_type'] == 'code']
si = {}
for i in code:
    for j in i['source']:
        if "#si-exercise" in j:
            exec(compile("".join(i['source']), '<string>', 'exec'))


class TestCase(unittest.TestCase):

    def testLogitCorrectValues(self):
        data = pd.read_csv("tests/files/assignment8Data.csv")
        x = data[['sex', 'age', 'educ']]
        y = data['white']

        reg = RegressionModel(x, y, create_intercept=True, regression_type='logit')
        reg.fit_model()

        print("\nCustom model results:")
        for key, value in reg.results.items():
            print(f"{key}: {value}")

        x_sm = sm.add_constant(x)
        model = Logit(y, x_sm)
        result = model.fit()

        print("\nStatsmodels summary:")
        print(result.summary())

        params = result.params
        bse = result.bse
        z_stats = result.tvalues
        p_values = result.pvalues

        print("\nExtracted values from statsmodels:")
        print(f"Params: {params}")
        print(f"Standard Errors: {bse}")
        print(f"Z-statistics: {z_stats}")
        print(f"P-values: {p_values}")

        sex = {
            'coefficient': params['sex'],
            'standard_error': bse['sex'],
            'z_stat': z_stats['sex'],
            'p_value': p_values['sex']
        }
        age = {
            'coefficient': params['age'],
            'standard_error': bse['age'],
            'z_stat': z_stats['age'],
            'p_value': p_values['age']
        }
        educ = {
            'coefficient': params['educ'],
            'standard_error': bse['educ'],
            'z_stat': z_stats['educ'],
            'p_value': p_values['educ']
        }
        intercept = {
            'coefficient': params['const'],
            'standard_error': bse['const'],
            'z_stat': z_stats['const'],
            'p_value': p_values['const']
        }

        tolerance = 1e-6

        sexEq = np.isclose(sex['coefficient'], reg.results['sex']['coefficient'], atol=tolerance) & \
                np.isclose(sex['standard_error'], reg.results['sex']['standard_error'], atol=tolerance) & \
                np.isclose(sex['z_stat'], reg.results['sex']['z_stat'], atol=tolerance) & \
                np.isclose(sex['p_value'], reg.results['sex']['p_value'], atol=tolerance)

        print(f"\nSex comparison: {sexEq}")

        ageEq = np.isclose(age['coefficient'], reg.results['age']['coefficient'], atol=tolerance) & \
                np.isclose(age['standard_error'], reg.results['age']['standard_error'], atol=tolerance) & \
                np.isclose(age['z_stat'], reg.results['age']['z_stat'], atol=tolerance) & \
                np.isclose(age['p_value'], reg.results['age']['p_value'], atol=tolerance)

        print(f"Age comparison: {ageEq}")

        educEq = np.isclose(educ['coefficient'], reg.results['educ']['coefficient'], atol=tolerance) & \
                 np.isclose(educ['standard_error'], reg.results['educ']['standard_error'], atol=tolerance) & \
                 np.isclose(educ['z_stat'], reg.results['educ']['z_stat'], atol=tolerance) & \
                 np.isclose(educ['p_value'], reg.results['educ']['p_value'], atol=tolerance)

        print(f"Educ comparison: {educEq}")

        interceptEq = np.isclose(intercept['coefficient'], reg.results['intercept']['coefficient'], atol=tolerance) & \
                      np.isclose(intercept['standard_error'], reg.results['intercept']['standard_error'], atol=tolerance) & \
                      np.isclose(intercept['z_stat'], reg.results['intercept']['z_stat'], atol=tolerance) & \
                      np.isclose(intercept['p_value'], reg.results['intercept']['p_value'], atol=tolerance)

        print(f"Intercept comparison: {interceptEq}")

        # Debugging intercepts in RegressionModel and statsmodels
        print("\nIntercept in Custom Model Results:")
        print(f"Coefficient: {reg.results['intercept']['coefficient']}")
        print(f"Standard Error: {reg.results['intercept']['standard_error']}")
        print(f"Z-Statistic: {reg.results['intercept']['z_stat']}")
        print(f"P-Value: {reg.results['intercept']['p_value']}")

        print("\nIntercept in Statsmodels Results:")
        print(f"Coefficient: {intercept['coefficient']}")
        print(f"Standard Error: {intercept['standard_error']}")
        print(f"Z-Statistic: {intercept['z_stat']}")
        print(f"P-Value: {intercept['p_value']}")


        self.assertTrue(sexEq & ageEq & educEq & interceptEq, "Your coefficients are not correctly calculated.")