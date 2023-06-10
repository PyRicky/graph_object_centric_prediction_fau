import pandas as pd
import sys, os, json
import time, datetime
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, mean_absolute_error


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def add_pseudo_activity(args, df):
    if args.add_pseudo_start:
        print('Adding pseudo start per case')
        case = -1
    else:
        print('Adding pseudo end activities per case')
        case = 0
    
    index = 0
    percentage = 0
    df_init_length = len(df)
    # treat the resource column as categorical
    if args.first_context_attr_name in df.columns:
        df[args.first_context_attr_name] = df[args.first_context_attr_name].astype('str')

    # this is needed in order to insert an empty row respecting column types
    # empty_row = []
    # for column_type in df.dtypes:
    #     if column_type == "object":
    #         empty_row.append('')
    #     else:
    #         empty_row.append(np.nan)

    if args.add_pseudo_start:
        # calculate pseudo starts (needed for predicting from prefixes with length 1)
        # the indexes are needed for placing back later the start pseudo activities
        df_pseudo_starts = df.reset_index().copy()
        # keep the first element of each case - the pseudo start has the same time characteristics of the first event of the case (it's a fake activity) - but no case ids
        df_pseudo_starts = df_pseudo_starts.groupby("Case ID").nth(0).reset_index()
        df_pseudo_starts = df_pseudo_starts.set_index("index")
        df_pseudo_starts.index = df_pseudo_starts.index - 0.3
        df_pseudo_starts[args.activity_key] = "pseudo_start"
        df_pseudo_starts[args.id_cols] = np.nan
        df = pd.concat([df, df_pseudo_starts])
        # reorder the dataframe and place the pseudo activities in the right places

    # while index < df_init_length:
        
    #     # if args.add_pseudo_end:
    #     #     if index + 1 == len(df) or (
    #     #             index + 1 < len(df) and df.iloc[index + 1, 0] != case and df.iloc[index, 0] != df.iloc[
    #     #         index + 1, 0]):  # end pseudo

    #     #         # made this more generic, for each number of possible columns
    #     #         # df.loc[(index + 0.6)] = case, 'pseudo_end', '', ''
    #     #         df.loc[(index + 0.6)] = empty_row
    #     #         df.loc[(index + 0.6), args.case_id_key] = case
    #     #         df.loc[(index + 0.6), args.activity_key] = 'pseudo_end'
    #     #         case += 1

    #     if int(index % (df_init_length / 10)) == 0:
    #         print("{} %".format(percentage))
    #         percentage += 10
    #     index += 1

    # cast case id column back to int and cast time column as a str
    # df[args.case_id_key] = df[args.case_id_key].astype("int")
    # df[args.time_key] = df[args.time_key].fillna('')
    return df.sort_index().reset_index(drop=True)

def add_time_features_to_dataframe(df, args):
    start_time = time.time()
    #add time features 
    # time from previous event
    df = df.assign(time_from_previous_event=df[args.time_key].shift(1))
    df.loc[df["time_from_previous_event"].isnull(), "time_from_previous_event"] = df.loc[df["time_from_previous_event"].isnull(), args.time_key]
    df["time_from_previous_event"] = round((df[args.time_key] - df["time_from_previous_event"]) / 86400, 4)
    # put to 0 the time for every first event of the case
    df.loc[~df[args.case_id_key].duplicated(), "time_from_previous_event"] = 0

    df = df.assign(time_from_start=df.groupby(args.case_id_key)[args.time_key].apply(lambda x: round((x - x.iloc[0]) / 86400, 4))) # time from start
    df = df.assign(time_from_midnight=pd.to_datetime(df[args.time_key], unit='s').apply(lambda x: round((x - x.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 86400, 4))) # time from midnight
    df = df.assign(weekday=pd.to_datetime(df[args.time_key], unit='s').apply(lambda x: x.weekday())) # weekday
    print(f"Time for calculating time features: {str(datetime.timedelta(seconds=round(time.time() - start_time)))}")
    return df


def select_path(args, df):
    """ Select only events in the path (from source to target)"""
    print("Selecting path ...")
    start_time = time.time()
    # df4 = pd.read_csv(args.csv.replace(".csv", "_concurrent.csv"), low_memory=False).copy()

    # not all path could contain the source activity or the target activity - discard those cases
    source_cases = df.loc[df[args.activity_key] == args.source_activity, args.case_id_key].unique()
    target_cases = df.loc[df[args.activity_key] == args.target_activity, args.case_id_key].unique()
    cases = source_cases[np.isin(source_cases, target_cases)]
    df = df.loc[df[args.case_id_key].isin(cases)]
    # cut events before start and after target
    df = df.groupby(args.case_id_key).apply(lambda x: x[(x.index >= x[x[args.activity_key] == args.source_activity].index[0]) &
        (x.index <= x[x[args.activity_key] == args.target_activity].index[-1])]).droplevel(0)

    #cut events to be aligned with original dataset
    # df4 = df4.loc[df4.index.isin(df.index)].reset_index(drop=True)

    # renamed_df = renamed_df.loc[renamed_df.index.isin(df.index)].reset_index(drop=True)
    df.reset_index(drop=True, inplace=True)
    # renamed_df.reset_index(drop=True, inplace=True)
    
    # df = replicate_case_level_attributes(args, df, source_entity)

    print(f"Time for selecting paths in dataset: {str(datetime.timedelta(seconds=round(time.time() - start_time)))}")
    return df

def calculate_source_entity(args):
    #select the source entity
    if "contract" in f"../data/{args.eventlog}":
        source_entity = "CONTRACT"
    elif "req" in f"../data/{args.eventlog}":
        source_entity = "REQ"
    elif "order" in f"../data/{args.eventlog}":
        source_entity = "ORDER"
    elif "receipt" in f"../data/{args.eventlog}":
        source_entity = "RECEIPT"
    elif "connected" in f"../data/{args.eventlog}":
        source_entity = "CONNECTED"
    else:
        source_entity = None
    return source_entity


def replicate_case_level_attributes(args, df, source_entity):
    for column in df.columns:
        if source_entity in column and "ID" not in column:
            if not (df.loc[df[column].notnull(), [args.case_id_key, column]].groupby(args.case_id_key).nunique()[column] > 1).any():
                df[column] = df.groupby(args.case_id_key)[column].ffill()
    return df


def calculate_target_kpi(args, df, renamed_df, object_case_ids):
    print("Calculating target kpi ...")
    start_time = time.time()
    kpi_in_column = False

    #understand the type of kpi (needed in particular for the attribute prediction)
    if args.kpi == "activity" or args.kpi == "next_activity":
            kpi_type = "Categorical"
    elif args.kpi in df.columns:
        if df[args.kpi].dtype == "object":
            kpi_type = "Categorical"
        else:
            kpi_type = "Numerical"
    else:
        kpi_type = "Numerical"

    # we associate the target path time per case - needed to know which events needs to be discarded because above the target (the process is finished)
    renamed_df = pd.merge(renamed_df, (round((renamed_df.groupby("case:concept:name")["time:timestamp"].max() - \
                                                renamed_df.groupby("case:concept:name")["time:timestamp"].min()) / (86400), 4))\
                                                .rename("path_time"), on="case:concept:name")

    # calculate kpi target 
    if args.kpi == "next_activity":
        target = pd.Series(
            renamed_df[renamed_df.columns[renamed_df.columns.str.startswith("concept:name")]].shift(-1).values.tolist(),
            name="y")
        renamed_df = pd.concat([renamed_df, target], axis=1)
        renamed_df = renamed_df.loc[renamed_df["time_from_start"] < renamed_df["path_time"]]
        del renamed_df["path_time"]
        print("Calculated target column for next activity prediction")

    elif args.kpi == "path_time":
        renamed_df.rename(columns={'path_time': 'y'}, inplace=True)
        # delete all events with time_from_start higher than the target (more final events could have the same timestamp
        renamed_df = renamed_df.loc[renamed_df["time_from_start"] < renamed_df["y"]]
        print("Calculated target column for path time prediction")

    elif args.kpi == "activity":
        # predict if a certain activity will be performed - set 1 for each case until that activity happens, the rest 0
        df["y"] = 0
        # we calculate the target on the original not encoded dataframe and then we assign the target column
        case_ids = df.loc[df[args.activity_key] == args.pred_act][args.case_id_key].unique()
        for case_id in case_ids:
            index = df.loc[(df[args.case_id_key] == case_id) & (df[args.activity_key] == args.pred_act)].index
            if len(index) == 1:
                index = index[0]
            else:
                # if activity is performed more than once, take only the last occurrence
                index = index[-1]
            # put 1 to all y_targets before that activity in the case
            df.loc[(df[args.case_id_key] == case_id) & (df.index < index), "y"] = 1
        
        renamed_df['y'] = df['y']
        renamed_df = renamed_df.loc[renamed_df["time_from_start"] < renamed_df["path_time"]]
        del renamed_df["path_time"]
        print(f"Calculated target column for prediction of {args.pred_act} occurrence")

    elif args.kpi in df.columns:
        kpi_in_column = True
        if kpi_type == "Categorical":
            # case very similar to activity - e.g. pay_type=late
            df["y"] = 0
            case_ids = df.loc[df[args.kpi] == args.pred_act][args.case_id_key].unique()
            for case_id in case_ids:
                index = df.loc[(df[args.case_id_key] == case_id) & (df[args.kpi] == args.pred_act)].index
                if len(index) == 1:
                    index = index[0]
                else:
                    # if activity is performed more than once, take only the last
                    index = index[-1]
                # put 1 to all y_targets before that activity in the case
                df.loc[(df[args.case_id_key] == case_id) & (df.index < index), "y"] = 1

            renamed_df['y'] = df['y']
            renamed_df = renamed_df.loc[renamed_df["time_from_start"] < renamed_df["path_time"]]
            del renamed_df["path_time"]
            print(f"Calculated target column for prediction of {args.kpi} occurrence with value {args.pred_act}")
        else:
            #calculate avg numerical target (e.g. for each case what is the average invoice delay - if any)
            #calculate avg delay considering only late invoices, otherwise 0 
            df_last_attribute = df.loc[df[args.kpi] > 0, [args.case_id_key, args.kpi]].groupby(args.case_id_key).mean().reset_index()
            target_column = df[args.case_id_key].map(df_last_attribute.set_index(args.case_id_key)[args.kpi])
            target_column = target_column.fillna(0)
            target_column.rename('y', inplace=True)

            renamed_df['y'] = target_column
            renamed_df = renamed_df.loc[renamed_df["time_from_start"] < renamed_df["path_time"]]
            del renamed_df["path_time"]
            print(f"Calculated target column for estimation of {args.kpi}")
    else:
        raise ("Kpi not implemented")

    # saving indexes could be needed later if we want to keep the same events for different baselines
    indexes = renamed_df.index
    df = df.loc[df.index.isin(indexes)].reset_index(drop=True)
    object_case_ids = object_case_ids.loc[object_case_ids.index.isin(indexes)].reset_index(drop=True)
    renamed_df.reset_index(drop=True, inplace=True)
    print(f"Time to calculate target kpi: {str(datetime.timedelta(seconds=round(time.time() - start_time)))}")
    return renamed_df, df, kpi_type, kpi_in_column, object_case_ids

def print_kpi_statistics(args, df, kpi_type):
    print(f"Dataframe shape: {df.shape}")
    if kpi_type == "Numerical":
        avg_value = round(df.groupby("case:concept:name")["y"].max().mean(), 2)
        std_value = round(df.groupby("case:concept:name")["y"].max().std(), 2) 
    if args.kpi == "path_time":
        print(f"Avg Completed Cases Path Time: {avg_value} days")
        print(f"Std Completed Cases Path Time: {std_value} days")
    elif args.kpi == "total_cost":
        print(f"Avg Completed Cases Cost: {avg_value} Euros")
        print(f"Std Completed Cases Cost: {avg_value} Euros")
    elif args.kpi in df.columns and kpi_type != "Categorical":
        print(f"Avg # of payment late days per case: {avg_value}")
        print(f"Std # of payment late days per case: {std_value} days")
    elif args.kpi == "activity":
        print(f"Number of events with target class 1 (Activity {args.pred_act} will be performed): {df[df['y'] == 1].shape[0]} / {df.shape[0]}" )
        print(f'Number of cases containing the target activity: {len(df.loc[df["y"] == 1, "case:concept:name"].unique())} / '
            f'{len(df["case:concept:name"].unique())}')
    elif args.kpi != "next_activity":
        print(f"Number of events with target class 1 (Attribute {args.kpi}={args.pred_act} will occur): {df[df['y'] == 1].shape[0]} / {df.shape[0]}" )
        print(f'Number of cases containing the target: {len(df.loc[df["y"] == 1, "case:concept:name"].unique())} / '
            f'{len(df["case:concept:name"].unique())}')


def train_test_split(df, experiment_folder):
    """
    :param df: the original dataframe without the end activities and with the information about the cases
    :param dataset: the encoded graphs, which will be used for training
    :output: indexes (prefixes) related to train / test / valid
    """

    # take the first prefix for each case
    first_prefixes = df.groupby('Case ID',as_index=False).nth(0)
    # drop from the dataframe the first prefix, that is the pseudo_start (to match the graphs)
    # df = pd.concat([df,first_prefixes]).astype('str').drop_duplicates(keep=False).reset_index(drop=True)
    df = df.loc[~df.index.isin(first_prefixes.index)].reset_index(drop=True)

    # try dividing train/test 80-20 instead
    # number_train_cases = round(len(df["Case ID"].unique()) * 0.8)

    second_quartile = len(df["Case ID"].unique()) / 2
    third_quartile = len(df["Case ID"].unique()) / 4 * 3
    number_train_cases = round((second_quartile + third_quartile) / 2)

    print("Preparing dataset for training")
    np.random.seed(6415)
    # select the cases and take the correspondent indexes in the dataset (each index is a graph representing a prefix)
    cases = df["Case ID"].unique()
    train_cases = np.random.choice(cases, size=number_train_cases, replace=False)
    # this passage is needed otherwise the order is different from catboost logic and therefore the validation set would be different
    train_cases = df.loc[df["Case ID"].isin(train_cases), "Case ID"].unique()
    np.random.seed(6415)
    valid_cases = np.random.choice(train_cases.tolist(), size=int(len(train_cases.tolist()) / 100 * 20), replace=False)

    train_without_valid_cases = train_cases[~np.isin(train_cases, valid_cases)]
    test_cases = cases[~np.isin(cases, train_cases)]

    data_train = df.loc[df["Case ID"].isin(train_without_valid_cases)].index.tolist()
    data_val = df.loc[df["Case ID"].isin(valid_cases)].index.tolist()
    data_test = df.loc[df["Case ID"].isin(test_cases)].index.tolist()

    # save train cases just to check in case that we are using the same cases for train
    info = {}
    info["train_cases"] = train_cases.tolist()
    with open(f"../{experiment_folder}/model/data_info.json", 'w') as j:
        json.dump(info, j)
    return data_train, data_val, data_test


def calculate_accuracy_per_prefix_size(accuracy_per_prefix_len, min_prefix_len, max_prefix_len, x_test, y_pred, y_true,
                                       k_fold):
    """

    :param args:
    :param accuracy_per_prefix_len: dictonary containing acc per prefix lengths accumulated over folds
    :param x_test: test samples (3 dimensional)
    :param y_pred: vecotr of predicted labels
    :param y_true: vecotr of true labels
    :param k_fold: fold k of CV
    :return: accuracy_per_prefix_len filled with {..., 'prefix_size': accuracy, ....}
    """

    y_prefix_size = {}
    # todo: cut out prefixes in x_test and locate them in their y_prediction and y_true
    for index, sample in enumerate(x_test):
        sample_zero_rows = np.where(np.all(np.isclose(sample, 0), axis=1))[0]
        prefix_len = len(sample) - len(sample_zero_rows)
        y_prefix_size[prefix_len] = [index] if prefix_len not in y_prefix_size else y_prefix_size[prefix_len] + [index]

    for prefix_len in range(min_prefix_len, max_prefix_len + 1):
        cut_len_y_true_subset = []
        cut_len_y_pred_subset = []
        if prefix_len in y_prefix_size:
            for index_in_y in y_prefix_size[prefix_len]:
                cut_len_y_true_subset.append(y_true[index_in_y])
                cut_len_y_pred_subset.append(y_pred[index_in_y])
            if prefix_len not in accuracy_per_prefix_len:
                accuracy_per_prefix_len[prefix_len] = [
                    accuracy_score(y_true=cut_len_y_true_subset, y_pred=cut_len_y_pred_subset)]
            else:
                accuracy_per_prefix_len[prefix_len] += [
                    accuracy_score(y_true=cut_len_y_true_subset, y_pred=cut_len_y_pred_subset)]

    return accuracy_per_prefix_len


def calculate_accuracy_per_prefix_size_gnns(accuracy_per_prefix_len, min_prefix_len, max_prefix_len, y_prefix_size,
                                            y_pred, y_true, k_fold):
    """

    :param args:
    :param accuracy_per_prefix_len: dictonary containing acc per prefix lengths accumulated over folds
    :param y_prefix_size: length of prefix per label in y
    :param y_pred: vecotr of predicted labels
    :param y_true: vecotr of true labels
    :param k_fold: fold k of CV
    :return: accuracy_per_prefix_len filled with {..., 'prefix_size': accuracy, ....}
    """

    for prefix_len in range(min_prefix_len, max_prefix_len + 1):
        cut_len_y_true_subset = []
        cut_len_y_pred_subset = []
        if prefix_len in y_prefix_size:
            for index_in_y in y_prefix_size[prefix_len]:
                cut_len_y_true_subset.append(y_true[index_in_y])
                cut_len_y_pred_subset.append(y_pred[index_in_y])
            if prefix_len not in accuracy_per_prefix_len:
                accuracy_per_prefix_len[prefix_len] = [
                    recall_score(y_true=cut_len_y_true_subset, y_pred=cut_len_y_pred_subset, average="weighted")]
            else:
                accuracy_per_prefix_len[prefix_len] += [
                    recall_score(y_true=cut_len_y_true_subset, y_pred=cut_len_y_pred_subset, average="weighted")]

    return accuracy_per_prefix_len


def plot_accuracy_per_prefix_size(args, accuracy_per_prefix_len):
    """

    :param args: dictonary containing acc per prefix lengths accumulated over folds
    :param accuracy_per_prefix_len:
    :return:
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    prefix_lengths = range(args.min_prefix_len, args.max_prefix_len + 1)
    mean_line = [np.mean(accuracy_per_prefix_len[c]) if c in accuracy_per_prefix_len else np.nan for c in
                 prefix_lengths]

    f = open('../result/accuracy_prefix_{}'.format(args.eventlog), "a")

    f.write("Event log %s -- Architecture %s: %s" % (args.eventlog, args.architecture, mean_line))
    f.close()

    min_line = [np.percentile(accuracy_per_prefix_len[c], 25) if c in accuracy_per_prefix_len else np.nan for c in
                prefix_lengths]

    max_line = [np.percentile(accuracy_per_prefix_len[c], 75) if c in accuracy_per_prefix_len else np.nan for c in
                prefix_lengths]
    ax.plot(prefix_lengths, mean_line)
    ax.fill_between(prefix_lengths, min_line, max_line, alpha=.2)
    # ax.set_title(r'$M_{%s}$' % target_activity_abbreviation, fontsize=30)
    ax.set_xlabel('Size of Process Instance Prefix', fontsize=20)
    ax.set_xticks(np.arange(2, args.max_prefix_len + 1, step=1))
    ax.set_ylabel('Accuracy', fontsize=20)
    ax.set_ylim(0.0, 1.0)
    plt.show()


def calc_mean_result_log_values(args, architecture):
    dataframe = None
    for k in range(args.num_folds):
        dataframe_kfold = pd.read_csv(
            '../result/result_log_{}_fold{}_{}'.format(architecture, k, args.eventlog), header=0,
            index_col=0, squeeze=True)
        if k == 0:
            dataframe = dataframe_kfold
        else:
            dataframe[['precision', 'recall', 'f1-score']] += dataframe_kfold[['precision', 'recall', 'f1-score']]

    dataframe[['precision', 'recall', 'f1-score']] /= args.num_folds
    dataframe.to_csv('../result/result_log_{}_meanValues_{}'.format(architecture, args.eventlog))

def plot_training_loss(args, kpi_in_column, history, kpi_type, experiment_folder):
    history_frame = pd.DataFrame.from_dict(history, orient='index')
    epochs = history_frame.index.tolist()
    train_loss = history_frame["loss"].tolist()
    valid_loss = history_frame["val_loss"].tolist()
    plt.plot(epochs, train_loss, label = "Train loss")
    plt.plot(epochs, valid_loss, label = "Valid loss")
    plt.xlabel("Epochs")
    if args.kpi == "activity" or (kpi_in_column and kpi_type == "Categorical"):
        plt.ylabel("Binary Crossentropy")
    elif args.kpi == "next_activity":
        plt.ylabel("Categorical Crossentropy")
    else:
        plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.savefig(f"../{experiment_folder}/result/training_loss.png", dpi=300, bbox_inches="tight")


def calculate_f1_score(prefix):
    return f1_score(prefix["TEST"], prefix["Prediction"])

def write_and_plot_test_error(df, y_pred, y_true, kpi_type, kpi, experiment_folder):

    # write also your predictions against the groundtruth to understand the error per prefix
    predictions = pd.Series(y_pred, name="Prediction")
    groundtruth = pd.Series(y_true, name="TEST")
    df = pd.concat([df.reset_index(drop=True), predictions, groundtruth], axis=1)
    df.to_csv(f"../{experiment_folder}/result/results.csv", index=False)
    
    if kpi != "next_activity":
        plt.clf()
        df["Prefix"] = df.groupby("Case ID").cumcount()+1
        if kpi_type == "Categorical":
            # when predicting categorical KPIs we use F1 score, while for numericals we use MAE error
            accuracy_per_prefix = df.groupby("Prefix").apply(calculate_f1_score)
        else:
            df["Error"] = np.abs(df["Prediction"] - df["TEST"])
            accuracy_per_prefix = df.groupby("Prefix")["Error"].mean()

        prefix_frequency = df.groupby("Prefix")["TEST"].count()

        # accuracy vs frequency per prefix
        plt.clf()
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
        plt.savefig(f"../{experiment_folder}/result/accuracy_prefix_frequency.png")
        print("Written and plotted test set accuracy per prefix")

def write_test_set_results(df, y_pred, y_true, kpi_type, kpi, experiment_folder):

    write_and_plot_test_error(df, y_pred, y_true, kpi_type, kpi, experiment_folder)

    if kpi_type == "Categorical":
        report = classification_report(y_true, y_pred, digits=4)
        print(report)
        with open(f'../{experiment_folder}/result/scores.json', 'w') as f:
            json.dump(classification_report(y_true, y_pred, digits=4, output_dict=True), f)
        if kpi != "next_activity":
            score = f1_score(y_true, y_pred)
            print(f'F1 score: {score}')
            f = open(f"../{experiment_folder}/result/scores.txt", "w")
            f.write(f'F1 score: {score}')
            f.close()
    else:
        mae = mean_absolute_error(y_true, y_pred)
        print(f'Prediction MAE is: {mae}')
        f = open(f"../{experiment_folder}/result/scores.txt", "w")
        f.write(f'MAE: {mae}')
        f.close()


def visualize_case_distribution_and_remove_skewed_cases(df, renamed_df, kpi_type, remove_skewed_cases=True):
    '''If in this path there are very infrequent cases lasting a lot of events discard them (keep only those with length in the 95 percentile)'''
    # understand the distribution of the cases
    df["Prefix"] = df.groupby("Case ID").cumcount()+1
    cases_length = df.groupby("Case ID")["Prefix"].max()
    distribution_of_cases_length = cases_length.reset_index().groupby("Prefix").count().reset_index()

    ax = distribution_of_cases_length.set_index("Prefix").plot(kind="bar", figsize=(40, 10), color="blue")
    ax.set_xlabel("Case Length (# Events)")
    ax.set_ylabel("# Cases")

    median_events = cases_length.median()
    mean_events = cases_length.mean()
    std_events = cases_length.std()
    if kpi_type != "Categorical":
        text = f"Mean events / case: {round(mean_events, 2)}\nMedian events / case: {median_events}\nStd_dev events / case: {std_events}\n" \
                f"Mean process duration: {round(renamed_df.groupby('case:concept:name').nth(0)['y'].mean(), 2)} days\n" \
                f"Median process duration: {round(renamed_df.groupby('case:concept:name').nth(0)['y'].median(), 2)} days\n" \
                f"Standard deviation process duration: {round(renamed_df.groupby('case:concept:name').nth(0)['y'].std(), 2)} days"
    else:
        text = f"Mean events / case: {round(mean_events, 2)}\nMedian events / case: {median_events}\nStd_dev events / case: {std_events}\n"
    print(text)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    from matplotlib import pyplot as plt
    plt.savefig(f"../experiment_files/result/distribution_of_cases_length.png")
    plt.clf()
    # exclude the long tail (very skewed long cases) by keeping values under the 95 percentile (so we keep cases whose length is under the found threshold)
    percentile_95 = np.percentile(cases_length, 95)
    del df["Prefix"]

    if remove_skewed_cases is True:
        print(f"Removing {len(cases_length[cases_length > percentile_95])} skewed very long cases out of {len(cases_length)} cases")
        # [print(f"Removed case with length: {length}") for length in cases_length[cases_length > percentile_95].values]
        cases_length = cases_length[cases_length <= percentile_95]
        df = df.loc[df["Case ID"].isin(cases_length.reset_index()["Case ID"].unique())]
        renamed_df = renamed_df.loc[renamed_df.index.isin(df.index)]

        df.reset_index(drop=True, inplace=True)
        renamed_df.reset_index(drop=True, inplace=True)

    return df, renamed_df