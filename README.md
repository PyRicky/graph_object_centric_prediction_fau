# Graph object centric prediction

### If you have problems installing with pip the packages in the requirements.txt you can follow these steps:
conda create -p ./env python=3.6.8 \
conda activate ./env \
conda install -c anaconda cudatoolkit=11.2 \
conda install -c anaconda cudnn=8.1 \
pip install tensorflow-gpu==2.6.2 \
pip install pip==20.0.1 \
pip install -r "requirements_alternative.txt"

### Before running the project append your project path to PYTHONPATH (to use absolute imports)
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/"

### Command example
conda activate ./env \
cd gocp/src 

##### 1st kpi: Next activity prediction #####
python main_mode_1_time_case.py --eventlog bpi2012_w_sample_raw.csv --first_context_attr_name resource --case_id_key case_id --activity_key activity \
python main_mode_1_time_case.py --eventlog df_connected_components.csv --first_context_attr_name USER --case_id_key "Case ID" --activity_key ACTIVITY_NAME --time_key START_DATE_TIME -s 'Contract Line Creation' -t 'Invoice Cleared' --id_cols ID_CONTRACT_LINE ID_REQ_LINE ID_ORDER_LINE ID_RECEIPT_LINE ID_INVOICE_HEADER -exp contract_to_inv_rec_next_act

##### 2nd kpi: Path Time prediction #####
python main_mode_1_time_case.py --eventlog bpi2012_w_sample_raw.csv --first_context_attr_name resource --case_id_key case_id --activity_key activity --kpi path_time \
python main_mode_1_time_case.py --eventlog df_connected_components.csv --first_context_attr_name USER --case_id_key "Case ID" --activity_key ACTIVITY_NAME --time_key START_DATE_TIME -s 'Contract Line Creation' -t 'Invoice Receipt' --id_cols ID_CONTRACT_LINE ID_REQ_LINE ID_ORDER_LINE ID_RECEIPT_LINE ID_INVOICE_HEADER --kpi path_time -exp contract_to_inv_rec_time

##### 2nd kpi: Path Time prediction (IT dataset) #####
python main_mode_1_time_case.py --eventlog df_connected_components_it.csv --first_context_attr_name USER --case_id_key "Case ID" --activity_key ACTIVITY_NAME --time_key START_DATE_TIME -s 'Purchase Requisition Line Created' -t 'Invoice Reconciled' --id_cols ID_REQ_LINE ID_ORDER_LINE ID_INVOICE_HEADER --kpi path_time -exp it_req_to_inv_recon


##### 3rd kpi: Activity occurrence prediction #####
python main_mode_1_time_case.py --eventlog bpi2012_w_sample_raw.csv --first_context_attr_name resource --case_id_key case_id --activity_key activity --kpi activity --pred_act 'W_Beoordelen fraude' \
python main_mode_1_time_case.py --eventlog df_connected_components.csv --first_context_attr_name USER --case_id_key "Case ID" --activity_key ACTIVITY_NAME --time_key START_DATE_TIME -s 'Contract Line Creation' -t 'Invoice Cleared' --id_cols ID_CONTRACT_LINE ID_REQ_LINE ID_ORDER_LINE ID_RECEIPT_LINE ID_INVOICE_HEADER --kpi activity --pred_act 'Purchase Order Blocked' -exp contract_to_inv_cl_po_blocked

#### 3rd kpi 2nd example: From Contract Creation to Invoice Cleared (activity Invoice Pay Method Changed) ###
python main_mode_1_time_case.py --eventlog df_connected_components.csv --first_context_attr_name USER --case_id_key "Case ID" --activity_key ACTIVITY_NAME --time_key START_DATE_TIME -s 'Contract Line Creation' -t 'Invoice Cleared' --id_cols ID_CONTRACT_LINE ID_REQ_LINE ID_ORDER_LINE ID_RECEIPT_LINE ID_INVOICE_HEADER --kpi 'activity' --pred_act 'Invoice Pay Method Changed' -exp contract_to_inv_cl_act_changed

##### 4th kpi: Prediction of occurrence of attributes with specific values #####
python main_mode_1_time_case.py --eventlog df_connected_components.csv --first_context_attr_name USER --case_id_key "Case ID" --activity_key ACTIVITY_NAME --time_key START_DATE_TIME -s 'Contract Line Creation' -t 'Invoice Cleared' --id_cols ID_CONTRACT_LINE ID_REQ_LINE ID_ORDER_LINE ID_RECEIPT_LINE ID_INVOICE_HEADER --kpi PAY_TYPE --pred_act 'LATE' -exp contract_to_inv_cl_pay_type_late

##### 5th kpi: Prediction of average value of numerical attributes #####
python main_mode_1_time_case.py --eventlog df_connected_components.csv --first_context_attr_name USER --case_id_key "Case ID" --activity_key ACTIVITY_NAME --time_key START_DATE_TIME -s 'Contract Line Creation' -t 'Invoice Cleared' --id_cols ID_CONTRACT_LINE ID_REQ_LINE ID_ORDER_LINE ID_RECEIPT_LINE ID_INVOICE_HEADER --kpi PAY_DELAY -exp contract_to_inv_cl_pay_delay


## TODOs
1) Generalize handling input dataset - <span style="color:blue">Riccardo</span> 🏆
2) Optimize computation preprocessing phase - <span style="color:blue">Riccardo</span> 🏆
3) Add Kpis to be predicted, both in command line and y target calculation (activity occurrence prediction or attribute occurring with certain values (binary 0 / 1), total time prediction, value estimation of numerical attributes) - <span style="color:blue">Riccardo</span> 🏆
4) Enable selection of paths for kpi calculation (default entire process) - <span style="color:blue">Riccardo</span> 🏆
5) Add multiple case ids as contextual attributes nodes - <span style="color:blue">Riccardo</span> 🏆
6) Modify edge types calculation for the object centric case (edge recursive or backward only if activity happened for the same multiple case ids) - Riccardo 🏆
7) Preprocess the data in order to clean the dataset (delete useless columns and reduce cardinality of categorical columns) - <span style="color:blue">Riccardo</span> 🏆
8) Understand and fix why the code is not working for the new dataset (tensorflow.python.framework.errors_impl.InvalidArgumentError: Paddings must be non-negative: 0 -124) - <span style="color:red">Sven</span> 🏆
9) Adapt graph architecture in order to predict different KPI types (binary classification / multiple classification / regression) - <span style="color:red">Riccardo</span> 🏆
10) Adapt the code and add different metrics for evaluations for the new different KPI types (e.g. the accuracy can't be used for regression) - <span style="color:red">Riccardo</span> 🏆 
11) Adapt validation test train split to test on the same exact data of Riccardo's paper - <span style="color:blue">Riccardo</span> 🏆
12) Avoid the graphs computation in memory - <span style="color:blue">Riccardo</span> 🏆
13) Solve the memory leak problem - <span style="color:blue">Riccardo</span> 🏆
14) Solve the model reloading problem - Riccardo & Sven 🏆
15) Start predicting from prefix with length 1 (add pseudo start) - <span style="color:blue">Riccardo</span> 🏆
16) Modify the adjacency matrix to connect two events only when there is a connection (one of the ids is in common) - <span style="color:blue">Riccardo </span> 🏆
17) Add procedure in order to create the event log from the original without considering a viewpoint - <span style="color:blue">Riccardo</span> 🏆
18) Add plots to better understand cases distribution, evaluate error Catboost vs GNN, and discard very long cases - <span style="color:blue">Riccardo</span> 🏆
19) Try different GNN architectures to increase the accuracy - <span style="color:red">Sven</span> 🏆
20) Try different encoding of features / try adding or removing features to increase the accuracy - Riccardo & Sven 🔨 
21) Run experiments on GCN - Sven 🔨
22) Create LSTM and run experiments on that - Riccardo 🏆
23) Adapt the logic to combine Catboost and the GNN based on the results on the validation set - Riccardo 🏆






