import pandas as pd
import scipy.stats as stats
import collections


def get_significance(pvalue):
    significance = ""
    if pvalue <= 0.05:
        significance += "*"
    if pvalue <= 0.01:
        significance += "*"
    if pvalue <= 0.001:
        significance += "*"
    return significance


def correlation_data(data_1, data_2, name_data_1, name_data_2):
    data_1_columns = data_1.columns
    data_2_columns = data_2.columns

    correlation_results = collections.defaultdict(lambda: [])
    correlation_matrix = pd.DataFrame(0, index=data_1_columns, columns=data_2_columns)
    significance_matrix = pd.DataFrame(0, index=data_1_columns, columns=data_2_columns)
    for col_1 in data_1_columns:
        for col_2 in data_2_columns:
            col_1_values = data_1[col_1].dropna()
            col_2_values = data_2[col_2].dropna()
            idx = col_1_values.index.intersection(col_2_values.index)
            corr, p_value = stats.pearsonr(col_1_values.loc[idx], col_2_values.loc[idx])
            correlation_results[name_data_1].append(col_1)
            correlation_results[name_data_2].append(col_2)
            correlation_results["Correlation"].append(corr)
            correlation_results["p-value"].append(p_value)
            correlation_results["Significance"].append(get_significance(p_value))
            correlation_matrix.loc[col_1, col_2] = corr
            significance_matrix.loc[col_1, col_2] = get_significance(p_value)
    correlation_results = dict(correlation_results)
    correlation_results = pd.DataFrame.from_dict(correlation_results)
    correlation_results = correlation_results.sort_values(by="p-value", axis=0, ascending=True)
    correlation_matrix = correlation_matrix.fillna(0)

    return correlation_results, correlation_matrix, significance_matrix


def rsquared_data(data_1, data_2, name_data_1, name_data_2):
    data_1_columns = data_1.columns
    data_2_columns = data_2.columns

    rsquared_results = collections.defaultdict(lambda: [])
    rsquared_matrix = pd.DataFrame(0, index=data_1_columns, columns=data_2_columns)
    significance_matrix = pd.DataFrame(0, index=data_1_columns, columns=data_2_columns)
    for col_1 in data_1_columns:
        for col_2 in data_2_columns:
            col_1_values = data_1[col_1].dropna()
            col_2_values = data_2[col_2].dropna()
            idx = col_1_values.index.intersection(col_2_values.index)
            slope, intercept, r_value, p_value, std_err = stats.linregress(col_1_values.loc[idx], col_2_values.loc[idx])
            rsquared = r_value**2
            rsquared_results[name_data_1].append(col_1)
            rsquared_results[name_data_2].append(col_2)
            rsquared_results["R^2"].append(rsquared)
            rsquared_results["p-value"].append(p_value)
            rsquared_results["Significance"].append(get_significance(p_value))
            rsquared_matrix.loc[col_1, col_2] = rsquared
            significance_matrix.loc[col_1, col_2] = "%s\n%s" % (round(rsquared,2), get_significance(p_value))
    rsquared_results = dict(rsquared_results)
    rsquared_results = pd.DataFrame.from_dict(rsquared_results)
    rsquared_results = rsquared_results.sort_values(by="p-value", axis=0, ascending=True)
    rsquared_matrix = rsquared_matrix.fillna(0)

    return rsquared_results, rsquared_matrix, significance_matrix


def ttest_data(data_1, data_2, name_data_1, name_data_2):
    data_1_columns = data_1.columns
    data_2_columns = data_2.columns

    ttest_results = collections.defaultdict(lambda: [])
    correlation_matrix = pd.DataFrame(0, index=data_1_columns, columns=data_2_columns)
    significance_matrix = pd.DataFrame(0, index=data_1_columns, columns=data_2_columns)
    for col_1 in data_1_columns:
        for col_2 in data_2_columns:
            col_1_values = data_1[col_1].dropna()
            col_2_values = data_2[col_2].dropna()
            idx = col_1_values.index.intersection(col_2_values.index)
            ttest, p_value = stats.ttest_ind(col_1_values.loc[idx], col_2_values.loc[idx])
            ttest_results[name_data_1].append(col_1)
            ttest_results[name_data_2].append(col_2)
            ttest_results["T-test"].append(ttest)
            ttest_results["p-value"].append(p_value)
            ttest_results["Significance"].append(get_significance(p_value))
            correlation_matrix.loc[col_1, col_2] = ttest
            significance_matrix.loc[col_1, col_2] = get_significance(p_value)
    correlation_results = dict(ttest_results)
    correlation_results = pd.DataFrame.from_dict(correlation_results)
    correlation_results = correlation_results.sort_values(by="p-value", axis=0, ascending=True)
    correlation_matrix = correlation_matrix.fillna(0)

    return correlation_results, correlation_matrix, significance_matrix

