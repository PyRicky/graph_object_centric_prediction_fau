import argparse
import os

# from gocp.src import main_mode_1_time_case


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# why do we need to add a false end_activity at the end of each case?

def load():
    parser = argparse.ArgumentParser()

    # architecture
    parser.add_argument('--architecture', default='GCN', type=str)
    # ECC (1),
    # GAT (2),
    # GatedGraph (3)
    # GNN (4),
    # GIN0 (5),
    # GCS (6),
    # GCN (7)

    # event log specific
    # bpi2012_w_raw.csv, resource
    # sepsis_raw.csv, org_group
    parser.add_argument('--first_context_attr_name', default='org_group', type=str)
    parser.add_argument('--eventlog', default="sepsis_sample_raw.csv", type=str)

    # prognn setting specific
    parser.add_argument('--add_pseudo_end', default=False, type=bool)  # pseudo end activity
    parser.add_argument('--add_pseudo_start', default=True, type=bool)  # pseudo start activity
    parser.add_argument('--shuffle', default=True, type=bool)  # shuffling cases

    # Architecutre specfic
    parser.add_argument('--learning_rate', default=1e-3, type=int)  # 1e-4; 1e-3; 1e-2;
    parser.add_argument('--epochs', default=150, type=int)  # there a rule of thumb to make it 10% of number of epoch.
    parser.add_argument('--es_patience', default=15, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    # pm4py (also for handling generic dataset / generic kpi)
    parser.add_argument('--case_id_key', default="case:concept:name", type=str)
    parser.add_argument('--activity_key', default="concept:name", type=str)
    parser.add_argument('--time_key', default="time", type=str)
    parser.add_argument('--time_format', default="%d.%m.%Y-%H:%M:%S",
                        type=str)  # if the column is in seconds the format will be automatically handled later
    parser.add_argument('--kpi', default="next_activity", type=str, help='Kpi to be selected (path_time / activity)')
    parser.add_argument('--pred_act', type=str, help='for activity kpi, select activity to be predicted')
    parser.add_argument('--source_activity', '-s', type=str, help='Start activity of the path')
    parser.add_argument('--target_activity', '-t', type=str, help='End activity of the path')
    parser.add_argument('--experiment_name', '-exp', type=str, help='Name to give to experiment to eventually save files')

    #for handling multiple case ids for the object centric dataset
    parser.add_argument('--id_cols', type=str, nargs='+', help='List of multiple case ID columns for object-centric datasets')

    # Parameters for validation
    parser.add_argument('--seed', default=True, type=str2bool)
    parser.add_argument('--seed_val', default=876, type=int)
    parser.add_argument('--num_folds', default=10, type=int)
    parser.add_argument('--train_size', default=2/3, type=float)
    parser.add_argument('--val_split', default=0.2, type=float)  # defines percentage of training set which is used for validation

    parser.add_argument('--batch_size_train', default=32, type=int)

    # pre-processing
    parser.add_argument('--encoding_num', default="min_max_norm", type=str)  # onehot or hash for numerical attributes
    parser.add_argument('--encoding_cat', default="onehot", type=str)  # onehot or hash for categorical attributes

    parser.add_argument('--min_prefix_len', default=2, type=int)
    parser.add_argument('--max_prefix_len', default=15, type=int)

    # Parameters for gpu processing
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 0 (do not use gpu), -1 (use gpu if available)

    return args
