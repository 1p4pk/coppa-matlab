%% 
%Import required files/functions/libraries
addpath(genpath('./data_in/'));
addpath(genpath('./libs/'));
addpath(genpath('./examples/'));
addpath(genpath('./coppa/'));

%% User Input
% State range
min_state = 6;
max_state = 20;
splitPercentage = 70; % Split Training Set
model = 'hmm'; %Options: 'hmm','pfa','dbn'
num_iter = 2; %number of times the model will be initialized with different random values to avoid local optimum
dataset = 'test'; %Options: 'sap','bpi2013','test'
learn_new_model = 'yes'; %Options: 'yes','no'

%Load data set
%Input Required: only discrete attributes (except timestamp)
if strcmp(dataset,'sap')
    filename = './example/sap/SAP_P2P_COPPA_FULL.csv'; 
    delimiter = ';'; 
    timestamp_format = 'yyyy-MM-dd HH:mm:ss.SSSSSSS'; 
    CaseID = 1; Activity = 2; Timestamp = 3;    
elseif strcmp(dataset,'bpi2013')
    filename = './example/bpi2013/VINST cases closed problems.csv';
    delimiter = ';'; 
    timestamp_format = 'yyyy-MM-dd''T''HH:mm:ssXXX'; 
    CaseID = 1; Activity = 3; Timestamp = 2;
else
    filename = './example/data.csv'; 
    delimiter = ';'; 
    timestamp_format = 'yyyy-MM-dd''T''HH:mm:ssXXX'; 
    CaseID = 1; Activity = 2; Timestamp = 3;
end

%Load and Prepare Data
[dataTraining dataTesting unique_values N] = prepare_data(filename, delimiter, timestamp_format,CaseID,Timestamp,Activity,splitPercentage, model); 
 %% 
%Define model and start learning
if strcmp(learn_new_model,'yes')
    [bestoverallbnet bestoverallstate] = stategrid_learning(model, N ,dataTraining,num_iter,min_state, max_state, unique_values);
    disp(['Best number of states was ' num2str(bestoverallstate) '.']);
    save_name = ['bestbnet_' model '_' dataset '.mat'];
    save(save_name,'bestoverallbnet');
else
    disp('Loading saved model');
    load_name = ['bestbnet_' model '_' dataset '.mat'];
end

%% Draw Model
%G = bestbnet.dag;
%draw_graph(G);
%% 

%Prediction
[pred rv acc] = prediction(bestoverallbnet, dataTesting);