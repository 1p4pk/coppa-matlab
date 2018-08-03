%% 
%Import required files/functions/libraries
addpath(genpath('./data_in/'));
addpath(genpath('./libs/'));
addpath(genpath('./examples/'));
addpath(genpath('./coppa/'));

%%
%Load data set
%Input Required: only discrete attributes (including timestamp!)
filename = './example/data.csv'; delimiter = ';'; timestamp_format = 'yyyy-MM-dd''T''HH:mm:ssXXX'; CaseID = 1; Activity = 2; Timestamp = 3;
%filename = './example/bpi2013/VINST cases closed problems_COPPA.csv';delimiter = ';'; timestamp_format = 'yyyy-MM-dd''T''HH:mm:ssXXX'; CaseID = 1; Activity = 2; Timestamp = 3;
%filename = './example/sap/SAP_P2P_COPPA_FULL.csv'; delimiter = ','; timestamp_format = 'yyyy-MM-dd HH:mm:ss.SSSSSSS'; CaseID = 1; Activity = 2; Timestamp = 3;
%filename = './example/bpi2012/bpi_2012.csv';delimiter = ','; timestamp_format = 'yyyy/MM/dd HH:mm:ss.SSS'; CaseID = 1; Activity = 2; Timestamp = 4;

%Load Data
data = import_csv(filename, delimiter); 
load('mapping_table.mat');

%Prepare Data
[dataTraining dataTesting ns datn] = prepare_data(data,timestamp_format,CaseID,Timestamp,Activity,70); 
 %% 
%Define model
N = datn -1 + 1; % number of variables in one time slice. datn + 1 (for hidden state) -1 (for case id column)
bnet = create_dbn(N,ns);
%% 

%bestbnet =  learning(bnet,N,dataTraining,10);

load('bestbnet_allHVs.mat')

G = bestbnet.dag;
%draw_graph(G);

%pred = prediction(bestbnet, dataTesting);

mapped_value = map_to_name(2,2);
