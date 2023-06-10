import numpy
import pandas as pd
import time, datetime
from pm4py.objects.conversion.log import converter as log_converter
from sklearn.model_selection import KFold, train_test_split
import category_encoders
import gocp.src.preprocessing.common as common


def get_attribute_data_type(attribute_column):
    """ Returns the data type of the passed attribute column 'num' for numerical and 'cat' for categorical """

    column_type = str(attribute_column.dtype)

    # TODO: and if the column is an int??
    if column_type.startswith('float'):
        attribute_type = 'num'
    else:
        attribute_type = 'cat'

    return attribute_type


def encode_df(df, args):
    print("Encoding attributes ...")
    start_time = time.time() 
    if args.first_context_attr_name in df.columns:
        # treat the resource column as categorical
        df[args.first_context_attr_name] = df[args.first_context_attr_name].astype('str')
    encoded_df = df.copy()
    columns_not_to_be_encoded = ["case:concept:name", "time:timestamp"]
    if args.id_cols:
        columns_not_to_be_encoded.extend(args.id_cols.copy())
    for column in df:
        if column not in columns_not_to_be_encoded:  # no encoding of case ids and timestamp attributes
            encoding_mode = get_encoding_mode(df[column].dtype, args)
            encoding_columns = encode_column(df, column, encoding_mode, args)
            if encoding_mode == args.encoding_num:
                encoded_df[column] = encoding_columns
            else:
                del encoded_df[column]
                encoded_df[encoding_columns.columns] = encoding_columns
    print(f"Time for encoding: {str(datetime.timedelta(seconds=round(time.time() - start_time)))}")
    return encoded_df


def encode_case_ids_as_node_features(df, activity_column, args):
    """ Encodes the multiple case ids as node features (with 1 / 0 depending if the case id is present or not) """

    # save the original case ids activity columns in case we want to understand the results better later
    result_cols = ["case:concept:name"]
    result_cols.extend(args.id_cols.copy())
    df_result_case_ids = df[result_cols].copy()
    df_result_case_ids[activity_column.name] = activity_column
    
    df_objects = df[args.id_cols].notnull().astype("int")
    object_names = [x.replace("ID_", "").replace("_LINE", "").replace("_HEADER", "") for x in df_objects.columns.tolist()]
    df_objects.set_axis(object_names, axis=1, inplace=True)
    df = pd.concat([df, df_objects], axis=1)
    #delete the original case ids
    df.drop(args.id_cols, axis=1, inplace=True)

    return df, df_result_case_ids, object_names


def get_encoding_mode(column_type, args):
    """ Returns the encoding method to be used for a given data type """

    if column_type == 'object':
        encoding_mode = args.encoding_cat
        if encoding_mode == 'hash':
            encoding_mode = 'onehot'
    else:
        encoding_mode = args.encoding_num

    return encoding_mode


def encode_column(df, column_name, mode, args):
    """ Returns columns containing encoded values for a given attribute column """

    if mode == 'min_max_norm':
        encoding_columns = apply_min_max_normalization(df, column_name)

    elif mode == 'onehot':
        encoding_columns = apply_one_hot_encoding(df, column_name)

    elif mode == 'hash':
        encoding_columns = apply_hash_encoding(df, column_name, args)
    else:  # no encoding
        encoding_columns = df[column_name]

    return encoding_columns


def apply_min_max_normalization(df, column_name):
    """ Normalizes a data frame column with min max normalization """

    # if the column contains only positive values than -1 could represent a nan
    # if the column has also negative values 0 could represent a non influence, i.e. a 0
    if df[column_name].isnull().values.any():
        if numpy.any((df[column_name] < 0)):
            print(f"Filling nan values with 0 for column {column_name}")
            column = df[column_name].fillna(0)    
        else:
            print(f"Filling nan values with -1 for column {column_name}")
            column = df[column_name].fillna(-1)
    else:
        column = df[column_name]
    # todo: min-max normalization is not performed for GNNs
    # encoded_column = (column - column.min()) / (column.max() - column.min())

    return column


def apply_one_hot_encoding(df, column_name):
    """ Encodes a data frame column with one hot encoding """

    onehot_encoder = category_encoders.OneHotEncoder(cols=[column_name])
    encoded_column = onehot_encoder.fit_transform(df[column_name])

    return encoded_column


def apply_hash_encoding(df, column_name, args):
    """ Encodes a data frame column with hash encoding """

    hash_encoder = category_encoders.HashingEncoder(cols=[column_name],
                                                    n_components=args.num_hash_output,
                                                    hash_method='md5')
    encoded_df = hash_encoder.fit_transform(df)
    encoded_column = encoded_df[encoded_df.columns[pd.Series(encoded_df.columns).str.startswith('col_')]]

    new_column_names = []
    for number in range(len(encoded_column.columns)):
        new_column_names.append(column_name + "_%d" % number)

    encoded_column = encoded_column.rename(columns=dict(zip(encoded_column.columns.tolist(), new_column_names)))

    return encoded_column

def get_activity_id_mapping(renamed_df):
    onehot_encoder = category_encoders.OneHotEncoder(cols=["concept:name"])
    encoded_column = onehot_encoder.fit_transform(renamed_df["concept:name"])

    activities = renamed_df["concept:name"].unique()
    ohe_values = encoded_column.drop_duplicates().values
    ohe_values_str = [str(x) for x in ohe_values.tolist()]
    ohe_mapping = dict(zip(ohe_values_str, activities.tolist()))

    activity_id_name_mapping = {}
    for i, activity in enumerate(activities):
        activity_id_name_mapping[i] = activity
    
    return ohe_mapping, activity_id_name_mapping


def clean_columns_and_reduce_categorical_domain_dynamic(df, args, entity, mode="train"):
    '''Does some heavy cleaning on the oject-centric dataset'''

    #delete these columns (end_date is empty and others are duplicated ids)
    columns = ["END_DATE_TIME", "ID_CONTRACT_HEADER", "ID_REQ_HEADER", "ID_ORDER_HEADER", "ID_RECEIPT_HEADER", "ID_INVOICE_LINE"]
    df.drop(columns, axis=1, inplace=True)
    #delete also these columns, they have a different name but they are duplicated
    # df.loc[~df["REQ_VENDOR_NAME"].notnull(), ["REQ_ID_VENDOR"]] = None
    # df.loc[~df["REQ_PLANT_NAME"].notnull(), ["REQ_PLANT"]] = None
    # df.loc[~df["RECEIPT_PLANT_NAME"].notnull(), ["RECEIPT_PLANT"]] = None
    # df.loc[~df["ORDER_PLANT_NAME"].notnull(), ["ORDER_PLANT"]] = None
    # df.loc[~df["ORDER_PURCH_ORG_NAME"].notnull(), ["ORDER_PURCH_ORG"]] = None
    # df.loc[~df["ORDER_PURCH_GROUP_NAME"].notnull(), ["ORDER_PURCH_GROUP"]] = None
    # df.loc[~df["ORDER_COMPANY_NAME"].notnull(), ["ORDER_COMPANY"]] = None
    # df.loc[~df["ORDER_VENDOR_NAME"].notnull(), ["ORDER_ID_VENDOR"]] = None
    # df.loc[~df["PAY_VENDOR_NAME"].notnull(), ["PAY_VENDOR"]] = None

    # then we take only the id column and we remove the anonymized column name
    [print(f"Deleting duplicated column {x}") for x in ["INVOICE_VENDOR_NAME", "ORDER_VENDOR_NAME", "ORDER_COMPANY_NAME", "ORDER_PURCH_GROUP_NAME",
             "ORDER_PURCH_ORG_NAME", "ORDER_PLANT_NAME", "INVOICE_COMPANY_NAME", "INVOICE_PLANT_NAME",
             "RECEIPT_PLANT_NAME", "INVOICE_COMPANY_NAME"]]
    if "INVOICE_VENDOR_NAME" in df.columns:
        df.drop(["INVOICE_VENDOR_NAME", "ORDER_VENDOR_NAME", "ORDER_COMPANY_NAME", "ORDER_PURCH_GROUP_NAME",
                "ORDER_PURCH_ORG_NAME", "ORDER_PLANT_NAME", "INVOICE_COMPANY_NAME", "INVOICE_PLANT_NAME",
                "RECEIPT_PLANT_NAME", "INVOICE_COMPANY_NAME"], axis=1, inplace=True)
        df.drop(["REQ_VENDOR_NAME", "REQ_PLANT_NAME"], axis=1, inplace=True)

    if "CONTRACT_COMPANY_NAME" in df.columns:
        # do this in the approach where you have also contract columns (remove also req_quantity)
        # df.loc[~df["CONTRACT_COMPANY_NAME"].notnull(), ["CONTRACT_COMPANY"]] = None
        # df.loc[~df["CONTRACT_PURCHASING_ORGANIZATION_NAME"].notnull(), ["CONTRACT_PURCHASING_ORGANIZATION"]] = None
        # df.loc[~df["CONTRACT_PURCHASING_GROUP_NAME"].notnull(), ["CONTRACT_PURCHASING_GROUP"]] = None
        # df.loc[~df["CONTRACT_PLANT_NAME"].notnull(), ["CONTRACT_PLANT"]] = None
        df.drop(["CONTRACT_COMPANY_NAME", "CONTRACT_PURCHASING_ORGANIZATION_NAME", "CONTRACT_PURCHASING_GROUP_NAME", "CONTRACT_PLANT_NAME", "REQ_QUANTITY"], axis=1, inplace=True)

    start_time = time.time()
    columns = [col for col in df.columns if col not in args.id_cols and col != args.case_id_key and col != "ID_INVOICE_HEADER"
               and col != args.time_key and col != args.activity_key]
    date_format = '%Y-%m-%d'
    for column in columns:
        # some quantity columns could have multiple separators (, and .) - clean them before
        if (df[df[column].astype('str').str.contains(',.', na=False)].shape[0] != 0) and \
                ("NAME" not in column):
            print(f"Cleaning multiple separators for column: {column}")
            try:
                # this means italian notation (you can just avoid "."" for thousands and just replace "," 
                # of decimals with . to cast to float)
                df[column] = df[column].str.replace(".", "").str.replace(",", ".").astype("float")
            except:
                # in case it is american notation ("." already indicates decimals)
                df[column] = df[column].str.replace(",", "").astype("float")   
        # remove attributes that have only one value or it is null:
        total_attributes = len(df[column].dropna().unique())
        if (total_attributes == 0 or total_attributes == 1) and mode == "train":
            print(f"Deleting column (Only one attribute or always nan): {column}")
            del df[column]
        # remove attributes that are mostly unique - the len(unique()) is almost equal (85%) to case ids
        elif ((round(len(df[args.case_id_key].unique()) / 100 * 85)) <= (len(df[column].unique()))) and df[column].dtype == "object" and mode == "train":
            print(f"Deleting column (The column is a sort of case id - mostly unique values): {column}")
            del df[column]
        # take all cases for which the attribute is always null, if they are >= 80% delete the attribute
        elif ((len(df[args.case_id_key].unique()) / 100 * 80) <= len(df.loc[~df[args.case_id_key].isin(
            df.loc[df[column].notnull(), args.case_id_key].unique()), args.case_id_key].unique())) and mode == "train":
            print(f"Deleting column (The column is missing in at least 80% of cases): {column}")
            del df[column]
        else:
            # date columns extract month and part of month
            if "DATE" in column:
                print(f"Transforming Date column: {column}")
                month_column_name = column.replace("DATE", "MONTH")
                day_column_name = column.replace("DATE", "DAY")
                part_of_month_column_name = column.replace("DATE", "PART_OF_MONTH")
                df[month_column_name] = pd.DatetimeIndex(pd.to_datetime(df[column], format=date_format, errors='coerce')).month
                df[day_column_name] = pd.DatetimeIndex(pd.to_datetime(df[column], format=date_format, errors='coerce')).day
                df.loc[(df[day_column_name].notnull()) & (df[day_column_name] > 15), part_of_month_column_name] = 2
                df.loc[(df[day_column_name].notnull()) & (df[day_column_name] <= 15), part_of_month_column_name] = 1

                #these columns are not int
                df[month_column_name].fillna("missing", inplace=True)
                df[part_of_month_column_name].fillna("missing", inplace=True)
                df.loc[df[month_column_name] != "missing", month_column_name] = df.loc[df[month_column_name] != "missing",
                                                                                       month_column_name].astype('int')
                df.loc[df[part_of_month_column_name] != "missing", part_of_month_column_name] = df.loc[
                    df[part_of_month_column_name] != "missing", part_of_month_column_name].astype('int')

                df.drop([day_column_name, column], axis=1, inplace=True)
            else:
                # do the propagation only if it is a case-level attribute and a source entity attribute
                # (you don't want to propagate for ex user or a order attribute) - one value per case excluding missing
                if entity in column:
                    if not (df.loc[df[column].notnull(), [args.case_id_key, column]].groupby(args.case_id_key).nunique()[column] > 1).any():
                        print(f"Forwarding case-level attribute {column}")
                        df[column] = df.groupby(args.case_id_key)[column].ffill()
                if column in ["CONTRACT_MATERIAL_CODE", "REQ_MATERIAL_CODE", "ORDER_MATERIAL_CODE"]:
                    # these 3 columns are not numbers, transform them in categories
                    df.loc[df[column].isnull(), column] = "missing"
                    df.loc[(df[column] != 'missing'), column] = df.loc[((df[column] != 'missing'), column)].astype("str").\
                        str.replace("\\.0", "")

    print(f"Finished cleaning useless columns - shape after cleaning {df.shape}")

    # reduce the domain of several attributes when it is too skewed (logarithmic distribution)
    columns_cat_domain_to_be_reduced = [column for column in df.columns if (len(df[column].unique()) > 50) and
                                        df[column].dtype == "object" and column not in args.id_cols
                                        and column != args.activity_key and column != args.case_id_key and column != "ID_INVOICE_HEADER"]
    print("Reducing categorical attributes cardinality")
    for column in columns_cat_domain_to_be_reduced:
        case_entity = args.case_id_key
        #calculate the distribution and frequency of attribute's values
        attribute_values_frequency = df.groupby(column)[case_entity].nunique().sort_values(ascending=False)

        print(f"# Values for attribute {column} - Before: {len(attribute_values_frequency)}")
        # exclude attribute's values appearing only a few times
        attribute_values_frequency = attribute_values_frequency[attribute_values_frequency > 3]

        # before applying the threshold remove all the missing values (they are not missing)
        attribute_values_frequency = attribute_values_frequency[~attribute_values_frequency.index.isin(["missing", ""])]

        # apply 80-20 threshold
        threshold_80_20 = attribute_values_frequency.sum() / 100 * 80
        df_frequency = pd.concat([attribute_values_frequency, attribute_values_frequency.cumsum().rename("cumsum")],
                                 axis=1)
        #subtract the current value from cumsum, in order to include also the last value (otherwise if you have like 45-40%, you discard the second value)
        df_frequency["cumsum"] = df_frequency["cumsum"] - df_frequency[args.case_id_key]
        df_frequency = df_frequency[df_frequency["cumsum"] < threshold_80_20]

        # delete columns that have still a huge cardinality
        if len(df_frequency) > 80 or len(df_frequency) == 0:
            print(f"# Values for attribute {column} - After: {len(df_frequency)} - still huge cardinality, deleting it\n"
                  f"   --------------------")
            del df[column]
            continue
        else:
            print(f"# Values for attribute {column} - After: {len(df_frequency)}\n   --------------------")

        df.loc[(~df[column].isin(df_frequency.reset_index()[column])) & (df[column] != '') &
               (df[column] != 'missing') & (df[column].notnull()), column] = 'other'

    print(f"Finished reducing categorical attributes cardinality - shape {df.shape}")
    print(f"Time for cleaning the dataset: {str(datetime.timedelta(seconds=round(time.time() - start_time)))}")
    return df




class Preprocessor(object):
    iteration_cross_validation = 0
    activity = {}
    context = {}

    args = None

    def __init__(self, args):
        self.activity = {
            'activity_ids_to_one_hot': {},
            'one_hot_to_activity_ids': {},
            'activity_name_to_activity_id': {},
            'activity_id_to_activity_name': {},
            'label_length': 0,
        }
        self.context = {
            'attributes': [],
            'attributes_mapping_name_to_one_hot': {},  # can be accessed by attribute name
            'attributes_mapping_one_hot_to_name': {},
            'encoding_lengths': []
        }
        self.args = args

    def get_preprocessed_event_log_and_save_encoding_mapping(self, args, df):
        """
        Saves mapping for the eventlog attributes (activity and context)
        Needs to be called first of all
        :param df: eventlog as a pandas dataframe
        :return:
        """

        start_time = time.time()
        df_enc = self.encode_data_and_save_encoding(args, df)  # df_copy already loaded
        print(f"Time for encode attributes (facultative): {str(datetime.timedelta(seconds=round(time.time() - start_time)))}")

        parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"}
        event_log = log_converter.apply(df_enc, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
        self.get_context_attributes(df)

        # TODO: why returning a processed event_log (and also encode it), but in the main script the initial df is used?
        return event_log

    def get_eventlog_without_encoding_in_pm4py(self, df):
        """
        Saves mapping for the eventlog attributes (activity and context)
        Needs to be called first of all
        :param df: eventlog as a pandas dataframe
        :return:
        """

        parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: self.args.case_id_key}
        event_log = log_converter.apply(df, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
        self.get_context_attributes(df)
        return event_log

    def get_context_attributes(self, df=None):
        """ Retrieves names of context attributes """

        if df is not None:
            attributes = df.columns.tolist()
            attributes.remove("case:concept:name")
            attributes.remove("concept:name")
            attributes.remove("time:timestamp")
            self.context['attributes'] = attributes
        else:
            return self.context['attributes']

    def encode_context_attribute(self, df, column_name):
        """ Encodes values of a context attribute for all events in an event log """

        # data_type = get_attribute_data_type(df[column_name])
        encoding_mode = self.get_encoding_mode(df[column_name].dtype)

        encoding_columns = self.encode_column(df, column_name, encoding_mode)

        if isinstance(encoding_columns, pd.DataFrame):
            self.set_length_of_context_encoding(len(encoding_columns.columns))
        elif isinstance(encoding_columns, pd.Series):
            self.set_length_of_context_encoding(1)

        df = self.transform_encoded_attribute_columns_to_single_column(encoding_columns, df, column_name)

        return df[column_name], encoding_mode

    def encode_data_and_save_encoding(self, args, df):
        """ Encodes an event log represented by a data frame """
        # TODO: I have the feeling these copies could be optimized for memory saving
        # this copy is needed in order to not encode the original dataframe (we will work on the original dataframe picking the saved encoding)
        df_copy = df.copy()

        common.llprint('Encode attributes ... \n')
        columns_not_to_be_encoded = ["case:concept:name", "time:timestamp"]
        if args.id_cols:
            columns_not_to_be_encoded.extend(args.id_cols.copy())

        encoded_df = pd.DataFrame(df_copy["case:concept:name"])  # create initial dataframe with case id attribute

        for column_name in df_copy:
            if column_name in columns_not_to_be_encoded:  # no encoding of case ids and timestamp attributes 
                encoded_df[column_name] = df_copy[column_name]
            else:
                if column_name == "concept:name":
                    self.activity['activity_name_to_activity_id'] = self.map_activity_name_to_activity_id(
                        df_copy[column_name])
                    self.activity['activity_id_to_activity_name'] = self.map_activity_id_to_activity_name()

                    # for each activity replace the name with the corresponding id
                    activities = df_copy[column_name].unique()
                    for activity_name in activities:
                        activity_id = self.activity['activity_name_to_activity_id'][activity_name]
                        df_copy.loc[df_copy[column_name] == activity_name, column_name] = activity_id

                        # you have only one activity column and for each element the one-hot encoded vector (e.g. [1,0,0]),
                    # not the classic one-hot encoding with a column for each activity and 0 or a 1 depending if the activity is present or not
                    encoded_column = self.encode_activities(df_copy.copy(), column_name)
                    self.save_mapping_one_hot_to_id(column_name, df_copy[column_name],
                                                    encoded_column)  # save Mapping of one hot activities to ids
                else:  # encode context attributes
                    # TODO: does not make any sense to pass the full dataset just to encode one single column!!
                    encoded_column, encoding_mode = self.encode_context_attribute(df_copy.copy(), column_name)

                    if encoding_mode == self.args.encoding_cat:
                        self.save_mapping_one_hot_to_id(column_name, df_copy[column_name], encoded_column)

                encoded_df = encoded_df.join(encoded_column)

        common.llprint('Attributes encoded \n')
        return encoded_df

    def get_encoding_mode(self, data_type):
        """ Returns the encoding method to be used for a given data type """

        if data_type == 'object':
            mode = self.args.encoding_cat
        else:
            mode = self.args.encoding_num
        return mode

    def encode_activities(self, df, column_name):
        """ Encodes activities for all events in an event log """

        encoding_mode = self.args.encoding_cat

        if encoding_mode == 'hash':
            encoding_mode = 'onehot'

        encoding_columns = self.encode_column(df, column_name, encoding_mode)

        if isinstance(encoding_columns, pd.DataFrame):
            self.set_length_of_activity_encoding(len(encoding_columns.columns))
        elif isinstance(encoding_columns, pd.Series):
            self.set_length_of_activity_encoding(1)

        df = self.transform_encoded_attribute_columns_to_single_column(encoding_columns, df, column_name)

        return df[column_name]

    # TODO: check if this step is unnecessary because it only needs to be done since naptf2.0 has a converted event log and xnap2.0 has a raw event log
    def map_activity_name_to_activity_id(self, df_column):
        unique_events = []
        for event in df_column:
            if event not in unique_events:
                unique_events.append(event)

        event_name_to_id = {}
        for i, c in enumerate(unique_events):
            event_name_to_id[c] = i

        return event_name_to_id

    def map_activity_id_to_activity_name(self):
        unique_activity_ids_map_to_activity_names = {}
        for activity_name in self.activity['activity_name_to_activity_id']:
            unique_activity_ids_map_to_activity_names[
                self.activity['activity_name_to_activity_id'][activity_name]] = activity_name

        return unique_activity_ids_map_to_activity_names

    def encode_column(self, df, column_name, mode):
        """ Returns columns containing encoded values for a given attribute column """

        if mode == 'min_max_norm':
            encoding_columns = self.apply_min_max_normalization(df, column_name)

        elif mode == 'onehot':
            encoding_columns = self.apply_one_hot_encoding(df, column_name)

        elif mode == 'hash':
            encoding_columns = self.apply_hash_encoding(df, column_name)
        else:  # no encoding
            encoding_columns = df[column_name]

        return encoding_columns

    def save_mapping_one_hot_to_id(self, column_name, attribute_column, encoded_column):
        """ Saves the mapping from one hot to its id/name """

        attribute_ids = attribute_column.values.tolist()
        # TODO: why is this needed?
        if attribute_column.dtype != "object":
            attribute_ids = [str(i) for i in attribute_ids]

        # dataframe containing unique associations activity id - onehot encoding (0 [1,0,0]
        #                                                                         1 [0,1,0])
        df_attribute_encoding = pd.concat([pd.Series(attribute_ids, name="attribute_ids"), encoded_column], axis=1)
        df_attribute_encoding = df_attribute_encoding[~df_attribute_encoding["attribute_ids"].duplicated()]
        # convert one-hot encoded vectors from list to tuples
        df_attribute_encoding[column_name] = df_attribute_encoding[column_name].apply(
            lambda x: tuple(x) if type(x) == list else (x,))
        # result is a dict like {0: (1,0,0), 1: (0,1,0), ...}
        if column_name == 'concept:name':
            self.activity['activity_ids_to_one_hot'] = dict(df_attribute_encoding.values)
            self.activity['one_hot_to_activity_ids'] = dict([(t[1], t[0]) for t in df_attribute_encoding.values])
        else:
            self.context['attributes_mapping_name_to_one_hot'][column_name] = dict(df_attribute_encoding.values)
            self.context['attributes_mapping_one_hot_to_name'][column_name] = dict(
                [(t[1], t[0]) for t in df_attribute_encoding.values])

    def get_context_attribute_encoding_length(self, context_attribute_name):
        """
        :param context_attribute_name:
        :return: amount of columns in one hot encoding if attribute type is categorial or 1 if type is numerical
        """
        if context_attribute_name not in self.context['attributes_mapping_one_hot_to_name']:
            return 1  # In this case attribute type is numerical
        else:
            return len(self.context['attributes_mapping_one_hot_to_name'][context_attribute_name])

    # todo: getter for mapping one hot to id and name
    def get_context_attribute_one_hot_by_value_name(self, context_attribute_name, value):
        """
        :param context_attribute_name:
        :param value:
        :return:
        """
        if context_attribute_name not in self.context['attributes_mapping_one_hot_to_name']:
            return None  # In this case attribute type is numerical
        else:
            return self.context['attributes_mapping_name_to_one_hot'][context_attribute_name][value]

    def transform_encoded_attribute_columns_to_single_column(self, encoded_columns, df, column_name):
        """ Transforms multiple columns (repr. encoded attribute) to a single column in a data frame """

        encoded_values_list = encoded_columns.values.tolist()
        df[column_name] = encoded_values_list

        return df

    def apply_min_max_normalization(self, df, column_name):
        """ Normalizes a data frame column with min max normalization """

        column = df[column_name].fillna(df[column_name].mean())
        # todo: min-max normalization is not performed for GNNs
        # encoded_column = (column - column.min()) / (column.max() - column.min())

        return column

    def apply_one_hot_encoding(self, df, column_name):
        """ Encodes a data frame column with one hot encoding """

        onehot_encoder = category_encoders.OneHotEncoder(cols=[column_name])
        encoded_column = onehot_encoder.fit_transform(df[column_name])

        return encoded_column

    def apply_hash_encoding(self, df, column_name):
        """ Encodes a data frame column with hash encoding """

        hash_encoder = category_encoders.HashingEncoder(cols=[column_name],
                                                        n_components=self.args.num_hash_output,
                                                        hash_method='md5')
        encoded_df = hash_encoder.fit_transform(df)
        encoded_column = encoded_df[encoded_df.columns[pd.Series(encoded_df.columns).str.startswith('col_')]]

        new_column_names = []
        for number in range(len(encoded_column.columns)):
            new_column_names.append(column_name + "_%d" % number)

        encoded_column = encoded_column.rename(columns=dict(zip(encoded_column.columns.tolist(), new_column_names)))

        return encoded_column

    def set_length_of_activity_encoding(self, num_columns):
        """ Save number of columns representing an encoded activity """
        self.activity['label_length'] = num_columns

    def set_length_of_context_encoding(self, num_columns):
        """ Save number of columns representing an encoded context attribute """
        self.context['encoding_lengths'].append(num_columns)

    def get_length_of_activity_label(self):
        """ Returns number of columns representing an encoded activity """
        return self.activity['label_length']

    def get_lengths_of_context_encoding(self):
        """ Returns number of columns representing an encoded context attribute """
        return self.context['encoding_lengths']

    def get_activity_id_from_one_hot(self, one_hot_encoding):
        """
        :param one_hot_encoding:
        :return: activity id
        """
        if isinstance(one_hot_encoding, list):
            return self.activity['one_hot_to_activity_ids'][tuple(one_hot_encoding)]
        else:
            if one_hot_encoding not in self.activity['one_hot_to_activity_ids']:
                return len(self.activity['one_hot_to_activity_ids'])
            else:
                return self.activity['one_hot_to_activity_ids'][one_hot_encoding]

    def get_context_attribute_name_from_one_hot(self, attribute_name, one_hot_encoding):
        """
        :param one_hot_encoding:
        :param attribute_name: column name of context attribute
        :return: context_attribute_name (within event)
        """
        if isinstance(one_hot_encoding, list):
            return self.context['attributes_mapping_one_hot_to_name'][attribute_name][tuple(one_hot_encoding)]
        else:
            if one_hot_encoding not in self.context['attributes_mapping_one_hot_to_name'][attribute_name]:
                return len(self.context['attributes_mapping_one_hot_to_name'][attribute_name])
            else:
                return self.context['attributes_mapping_one_hot_to_name'][attribute_name][one_hot_encoding]

    def get_num_activities(self):
        """ Returns the number of activities (incl. artificial end activity) occurring in the event log """

        return len(self.activity['activity_ids_to_one_hot'])

    def context_exists(self):
        """ Checks whether context attributes exist """

        return len(self.get_context_attributes(df=None)) > 0

    def get_num_features(self):
        """ Returns the number of features used to train and test the model """

        num_features = 0
        num_features += self.get_length_of_activity_label()
        for len in self.get_lengths_of_context_encoding():
            num_features += len

        return num_features

    def get_num_context_attributes(self):
        """ returns the amount of context attributes """
        return len(self.context['attributes'])

    def get_max_case_length(self, event_log):
        """ Returns the length of the longest case in an event log """

        max_case_length = 0
        for case in event_log:
            if case.__len__() > max_case_length:
                max_case_length = case.__len__()

        return max_case_length

    def get_indices_split_validation(self, event_log):
        """ Produces indices for training and test set of a split-validation """

        indices_ = [index for index in range(0, len(event_log))]

        if self.args.shuffle:

            if self.args.seed:
                train_indices, test_indices, train_indices_, test_indices_ = train_test_split(indices_, indices_,
                                                                                              train_size=self.args.split_rate_train,
                                                                                              shuffle=self.args.shuffle,
                                                                                              random_state=self.args.seed_val)
                return train_indices, test_indices
            else:
                train_indices, test_indices, train_indices_, test_indices_ = train_test_split(indices_, indices_,
                                                                                              train_size=self.args.split_rate_train,
                                                                                              shuffle=self.args.shuffle,
                                                                                              random_state=None)
                return train_indices, test_indices

        else:

            return indices_[:int(len(indices_) * self.args.split_rate_train)], \
                   indices_[int(len(indices_) * self.args.split_rate_train):]

    def get_indices_k_fold_validation(self, event_log):
        """
        Produces indices for each fold of a k-fold cross-validation
        :param args:
        :param event_log:
        :return:
        """

        kFold = KFold(n_splits=self.args.num_folds, random_state=self.args.seed_val, shuffle=self.args.shuffle)

        train_index_per_fold = []
        test_index_per_fold = []

        for train_indices, test_indices in kFold.split(event_log):
            train_index_per_fold.append(train_indices)
            test_index_per_fold.append(test_indices)

        return train_index_per_fold, test_index_per_fold

    def get_cases_of_fold(self, event_log, index_per_fold):
        """ Retrieves cases of a fold """

        cases_of_fold = []

        for index in index_per_fold[self.iteration_cross_validation]:
            cases_of_fold.append(event_log[index])

        return cases_of_fold

    def get_subsequences_of_cases(self, cases):
        """ Creates subsequences of cases representing increasing prefix sizes """

        subseq = []

        for case in cases:
            for idx_event in range(0, len(case._list)):
                if idx_event == 0:
                    continue
                else:
                    subseq.append(case._list[0:idx_event])

        return subseq

    def get_next_events_of_subsequences_of_cases(self, cases):
        """ Retrieves next events (= suffix) following a subsequence of a case (= prefix) """

        next_events = []

        for case in cases:
            for idx_event in range(0, len(case._list)):
                if idx_event == 0:
                    continue
                else:
                    next_events.append(case._list[idx_event])

        return next_events

    def get_features_tensor(self, mode, event_log, subseq_cases):
        """ Produces a vector-oriented representation of feature data as a 3-dimensional tensor """

        num_features = self.get_num_features()
        max_case_length = self.get_max_case_length(event_log)

        if mode == 'train':
            features_tensor = numpy.zeros((len(subseq_cases),
                                           max_case_length,
                                           num_features), dtype=numpy.float64)
        else:
            features_tensor = numpy.zeros((1,
                                           max_case_length,
                                           num_features), dtype=numpy.float32)

        for idx_subseq, subseq in enumerate(subseq_cases):
            left_pad = max_case_length - len(subseq)

            for timestep, event in enumerate(subseq):

                # activity
                activity_values = event.get(self.args.activity_key)
                for idx, val in enumerate(activity_values):
                    features_tensor[idx_subseq, timestep + left_pad, idx] = val

                # context
                if self.context_exists():
                    start_idx = 0

                    for attribute_idx, attribute_key in enumerate(self.context['attributes']):
                        attribute_values = event.get(attribute_key)

                        if not isinstance(attribute_values, list):
                            features_tensor[
                                idx_subseq, timestep + left_pad, start_idx + self.get_length_of_activity_label()] = attribute_values
                            start_idx += 1
                        else:
                            for idx, val in enumerate(attribute_values, start=start_idx):
                                features_tensor[
                                    idx_subseq, timestep + left_pad, idx + self.get_length_of_activity_label()] = val
                            start_idx += len(attribute_values)

        return features_tensor

    def get_labels_tensor(self, subseq_cases):
        """ Produces a vector-oriented representation of labels as a 2-dimensional tensor from an event log snippet grouped by cases """

        num_event_labels = self.get_num_activities()

        labels_tensor = numpy.zeros((len(subseq_cases), num_event_labels), dtype=numpy.float64)

        for i, case in enumerate(subseq_cases):
            labels_tensor[i] = case[self.args.activity_key]

        return labels_tensor

    def get_predicted_label(self, predictions):
        """ Returns label of a predicted activity """

        predictions = predictions.tolist()
        prediction_idx = numpy.argmax(predictions)
        predictions_ = [0] * len(predictions)
        predictions_[prediction_idx] = 1

        return predictions_

    def convert_process_instance_list_id_to_name(self, process_instances):
        """
        takes a 2 dimensional input (meaning a list of process instances with their event ids and
        converts it to event names
        """
        process_instances_converted = []
        for process_instance in process_instances:
            process_instance_converted = []
            for event in process_instance:
                process_instance_converted.append(self.activity['activity_id_to_activity_name'][event])
            process_instances_converted.append(process_instance_converted)

        return process_instances_converted

    def get_random_process_instance(self, event_log, lower_bound, upper_bound):
        """
        Selects a random process instance from the complete event log.
        :param event_log:
        :param lower_bound:
        :param upper_bound:
        :return: process instance.
        """

        while True:
            x = len(event_log)
            rand = numpy.random.randint(x)
            size = len(event_log[rand])

            if lower_bound <= size <= upper_bound:
                break

        return event_log[rand]

    def get_cropped_instance_label(self, prefix_size, process_instance):
        """
        Crops the next activity label out of a single process instance.
        :param prefix_size:
        :param process_instance:
        :return:
        """

        if prefix_size == len(process_instance) - 1:
            # end marker
            return self.get_end_char()
        else:
            # label of next act
            return process_instance[prefix_size]

    # todo noticed that naming of event_id_to_name is still wrong and should be activity_id_to_name
    def get_activity_name_from_activity_id(self, activity_id):
        """
        :param activity_id:
        :return: activity name
        """
        if activity_id == len(self.activity['activity_id_to_activity_name']):
            return self.get_end_char()
        else:
            return self.activity['activity_id_to_activity_name'][activity_id]

    def get_activity_id_from_activity_name(self, activity_name):
        """
        :param activity_name:
        :return: activity id
        """

        return self.activity['activity_name_to_activity_id'][activity_name]

    def activity_contains_name(self, activity_name):
        return activity_name in self.activity['activity_name_to_activity_id']

    def get_activity_one_hot_from_activity_name(self, activity_name):
        """
        :param activity_name:
        :return:
        """
        activity_id = self.get_activity_id_from_activity_name(activity_name)

        return self.get_activity_one_hot_from_id(activity_id)

    def get_activity_one_hot_from_id(self, activity_id):
        """
        :param activity_id
        """

        return self.activity['activity_ids_to_one_hot'][activity_id]

    def get_activity_name_from_one_hot(self, activity_one_hot):
        return self.get_activity_name_from_activity_id(self.get_activity_id_from_one_hot(activity_one_hot))
