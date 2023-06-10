import os
# from pm4py.objects.log.adapters.pandas import csv_import_adapter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from collections import OrderedDict
import pandas as pd
import numpy as np
import sys
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petrinet import visualizer as pn_vis_factory

np.set_printoptions(threshold=sys.maxsize)


class Encoding:

    def get_event_log(self, log_csv):
        # get event log
        # eventlog_csv = csv_import_adapter.import_dataframe_from_path(os.path.join(self.path_dir), sep=";")
        log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)

        return self.convert_csv_to_event_log(log_csv)

    def rename_csv_columns(self, args, log_csv):

        log_csv = log_csv.rename(columns={args.case_id_key: 'case:concept:name', args.activity_key: 'concept:name', args.time_key: 'time:timestamp'})
        log_csv['concept:name'] = log_csv['concept:name'].astype(dtype=str)

        return log_csv

    def convert_csv_to_event_log(self, log_csv):
        param = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'}

        event_log = log_converter.apply(log_csv, parameters=param)
        return event_log
