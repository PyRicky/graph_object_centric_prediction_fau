import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

def calculate_f1_score(prefix):
    return f1_score(prefix["TEST"], prefix["Prediction"])

'''Just replace here the name of the experiment and put catboost results file in the gocp folder'''
# plot a line going to -1 or to 1 when graph wins against catboost per prefix

kpi_type = "Numerical"
experiments_folder = "experiments/final_experiments_new_valid"
experiment_name = "contract_to_inv_rec_time"
catboost_results_completed = "results_conmpleted_4_connected.csv"
catboost_results_valid = "results_valid_4_connected.csv"

df = pd.read_csv(f"../{experiments_folder}/{experiment_name}/result/results.csv")
df_catboost = pd.read_csv(f"../{catboost_results_completed}")

df["Prefix"] = df.groupby("Case ID").cumcount()+1
df_catboost["Prefix"] = df_catboost.groupby("Case ID").cumcount()+1

if kpi_type == "Categorical":
    # when predicting categorical KPIs we use F1 score, while for numericals we use MAE error
    accuracy_per_prefix = df.groupby("Prefix").apply(calculate_f1_score)
    accuracy_per_prefix_catboost = df_catboost.groupby("Prefix").apply(calculate_f1_score)

    # if positive value then catboost has a higher F1 (better), and we will have a point per catboost (corresponding to -1)
    comparison = accuracy_per_prefix = accuracy_per_prefix_catboost
    comparison[comparison > 0] = -1
else:
    df["Error"] = np.abs(df["Prediction"] - df["TEST"])
    accuracy_per_prefix = df.groupby("Prefix")["Error"].mean()
    df_catboost["Error"] = np.abs(df_catboost["path_time"] - df_catboost["TEST"])
    accuracy_per_prefix_catboost = df_catboost.groupby("Prefix")["Error"].mean()    

    # if negative value then catboost has a lower MAE (better), and we will have a point per catboost (corresponding to -1)
    comparison = accuracy_per_prefix_catboost - accuracy_per_prefix
    comparison[comparison < 0] = -1
    comparison[comparison > 0] = +1

plt.plot(comparison)
plt.xlabel("Prefix")
plt.ylabel("Models")
plt.yticks([-1, +1], ["Catboost", "GNN"])
plt.savefig(f"../{experiments_folder}/{experiment_name}/result/graph_catboost_comparison.png")



def write_and_plot_test_error(df_catboost, accuracy_per_prefix_catboost):
    '''This function is needed in case you had old experiments without the plots,
       for plotting accuracy of GNN vs Catboost,
       and for understanding if the accuracy of the results improves by removing very long cases (this won't be needed if we remove those cases directly in the training)
    '''

    df = pd.read_csv(f"../{experiments_folder}/{experiment_name}/result/results.csv")
    df["Prefix"] = df.groupby("Case ID").cumcount()+1
    if kpi_type == "Categorical":
        # when predicting categorical KPIs we use F1 score, while for numericals we use MAE error
        accuracy_per_prefix = df.groupby("Prefix").apply(calculate_f1_score)
    else:
        df["Error"] = np.abs(df["Prediction"] - df["TEST"])
        accuracy_per_prefix = df.groupby("Prefix")["Error"].mean()
    
    # accuracy vs frequency per prefix
    plt.clf()
    prefix_frequency = df.groupby("Prefix")["TEST"].count()
    fig, ax = plt.subplots()
    if kpi_type == "Categorical":
        ax.plot(accuracy_per_prefix, color='red', label="Accuracy")
    else:
        ax.plot(accuracy_per_prefix, color='red', label="Error")
    if kpi_type == "Categorical":
        ax.set_ylabel('F1')
    else:
        ax.set_ylabel('MAE')
    ax2 = ax.twinx()
    ax2.plot(prefix_frequency, color='blue', label="Frequency")
    ax2.set_ylabel('Frequency')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Prefix')
    plt.savefig(f"../{experiments_folder}/{experiment_name}/result/accuracy_prefix_frequency.png")

    plt.clf()
    # understand the distribution of the cases (in the test set)
    cases_length = df.groupby("Case ID")["Prefix"].max()
    distribution_of_cases_length = cases_length.reset_index().groupby("Prefix").count().reset_index()

    ax = distribution_of_cases_length.set_index("Prefix").plot(kind="bar", figsize=(40, 10), color="blue")
    ax.set_xlabel("Case Length (# Events)")
    ax.set_ylabel("# Cases")

    median_events = cases_length.median()
    mean_events = cases_length.mean()
    std_events = cases_length.std()
    text = f"Mean events / case: {round(mean_events, 2)}\nMedian events / case: {median_events}\nStd_dev events / case: {std_events}\n" \
           f"Mean process duration: {round(df.groupby('Case ID').nth(0)['TEST'].mean(), 2)} days\n" \
           f"Median process duration: {round(df.groupby('Case ID').nth(0)['TEST'].median(), 2)} days\n" \
           f"Standard deviation process duration: {round(df.groupby('Case ID').nth(0)['TEST'].std(), 2)} days"
    print(text)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.savefig(f"../{experiments_folder}/{experiment_name}/result/distribution_of_cases_length.png")
    # exclude the long tail (very skewed long cases) by keeping values under the 95 percentile (so we keep cases whose length is under the found threshold)
    percentile_95 = np.percentile(cases_length, 95)
    cases_length = cases_length[cases_length <= percentile_95]
    df_filtered = df.loc[df["Case ID"].isin(cases_length.reset_index()["Case ID"].unique())]
    df_catboost_filtered = df_catboost.loc[df_catboost["Case ID"].isin(cases_length.reset_index()["Case ID"].unique())]
    print(f"Threshold of cases length with 95 percentile: {percentile_95}")

    plt.clf()
    # let's compare accuracy before and after filtering, comparing GNN and Catboost
    accuracy_per_prefix = df_filtered.groupby("Prefix")["Error"].mean()
    accuracy_per_prefix_catboost = df_catboost_filtered.groupby("Prefix")["Error"].mean()
    prefix_frequency = df_filtered.groupby("Prefix")["TEST"].count()

    # let's keep only 95% of the observations for the evaluation of the error (since there are prefixes with a very low frequency) - OLD LOGIC
    # plt.clf()
    # indexes = prefix_frequency[prefix_frequency >= int(prefix_frequency.mean() + 2*prefix_frequency.std())].index
    # accuracy_per_prefix = accuracy_per_prefix.loc[indexes]
    # prefix_frequency = prefix_frequency.loc[indexes]

    fig, ax = plt.subplots()
    if kpi_type == "Categorical":
        ax.plot(accuracy_per_prefix, color='red', label="Accuracy")
    else:
        ax.plot(accuracy_per_prefix, color='red', label="Error")
    if kpi_type == "Categorical":
        ax.set_ylabel('F1')
    else:
        ax.set_ylabel('MAE')
    ax2 = ax.twinx()
    ax2.plot(prefix_frequency, color='blue', label="Frequency")
    ax2.set_ylabel('Frequency')
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Prefix')
    plt.savefig(f"../{experiments_folder}/{experiment_name}/result/accuracy_prefix_frequency_95.png")

    # compare with catboost on the reduced set of observations
    plt.clf()
    # accuracy_per_prefix_catboost = accuracy_per_prefix_catboost.loc[indexes]
    comparison = accuracy_per_prefix_catboost - accuracy_per_prefix
    # ax = comparison.plot(kind="bar", figsize=(40, 10), color="blue")

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(15, 15))
    y_pos = comparison.index
    values = comparison.values
    # TODO: adapt this also for the categorical case (the scale is inverted, the higher the F1 the better)
    ix_pos = values > 0
    ax.bar(y_pos[ix_pos], values[ix_pos], align='center', color='green')
    ax.bar(y_pos[~ix_pos], values[~ix_pos], align='center', color='b')

    ax.set_xlabel("Prefix")
    ax.set_ylabel("MAE")
    plt.savefig(f"../{experiments_folder}/{experiment_name}/result/graph_catboost_comparison_95_histogram.png")

    # if we exclude skewed prefixes from the dataset, how the accuracy changes?
    print(f'Catboost Error: {df_catboost["Error"].mean()}')
    print(f'GNN Error: {df["Error"].mean()}')
    print(f'Catboost Error without very skewed long cases: {df_catboost_filtered["Error"].mean()}')
    print(f'GNN Error without very skewed long cases: {df_filtered["Error"].mean()}')
    # print(f'Catboost Error without very skewed prefixes: {df_catboost[df_catboost["Prefix"].isin(indexes)]["Error"].mean()}')
    # print(f'GNN Error without very skewed prefixes: {df[df["Prefix"].isin(indexes)]["Error"].mean()}')

    print("Written and plotted test set accuracy per prefix")


def calculate_test_accuracy_combining_catboost_and_gnn():
    # Print the test loss combining the best models seen per prefix on the validation set
    # put catboost results on the gocp folder in order to calculate accuracy by combining the two models (based on performances on the validation set)
    df_results_gnn = pd.read_csv(f"../{experiments_folder}/{experiment_name}/result/results.csv")
    df_results_catboost = pd.read_csv(f"../{catboost_results_completed}")

    df_results_gnn["Prefix"] = df_results_gnn.groupby("Case ID").cumcount()+1
    df_results_catboost["Prefix"] = df_results_catboost.groupby("Case ID").cumcount()+1

    df_valid_gnn = pd.read_csv(f"../{experiments_folder}/{experiment_name}/result/results_valid.csv")
    df_valid_catboost = pd.read_csv(f"../{catboost_results_valid}")

    df_valid_gnn["Error"] = np.abs(df_valid_gnn["Prediction"] - df_valid_gnn["TEST"])
    df_valid_catboost["Error"] = np.abs(df_valid_catboost["Prediction"] - df_valid_catboost["TEST"])

    accuracy_per_prefix = df_valid_gnn.groupby("Prefix")["Error"].mean()
    accuracy_per_prefix_catboost = df_valid_catboost.groupby("Prefix")["Error"].mean()
    # catboost kpi was cast to second, convert it back to days
    accuracy_per_prefix_catboost = accuracy_per_prefix_catboost / (3600*24)
    df_results_catboost.rename(columns={df_results_catboost.columns[-3]: "Prediction"}, inplace=True)
    
    if kpi_type != "Categorical":
        comparison = accuracy_per_prefix_catboost - accuracy_per_prefix
    else:
        # in case the kpi is categorical the scale is inverted (the higher the better)
        comparison = accuracy_per_prefix - accuracy_per_prefix_catboost
    
    # we split prefixes in which is better catboost or the gnn
    catboost_prefixes = comparison[comparison < 0].index
    gnn_prefixes = comparison[comparison > 0].index

    # when gnn is better, the prediction will be made by the gnn
    print(f"MAE Catboost: {np.abs(df_results_catboost['Prediction'] - df_results_catboost['TEST']).mean()}")
    df_results_catboost.loc[df_results_catboost["Prefix"].isin(gnn_prefixes), "Prediction"] = df_results_gnn.loc[df_results_gnn["Prefix"].isin(gnn_prefixes), "Prediction"]
    print(f"MAE combining Catboost and GNN model: {np.abs(df_results_catboost['Prediction'] - df_results_catboost['TEST']).mean()}")


write_and_plot_test_error(df_catboost, accuracy_per_prefix_catboost)
calculate_test_accuracy_combining_catboost_and_gnn()