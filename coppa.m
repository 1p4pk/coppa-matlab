%% Import required files/functions/libraries
addpath(genpath('./data_in/'));
addpath(genpath('./libs/'));
addpath(genpath('./examples/'));
addpath(genpath('./coppa/'));

%% User Input
% State range
min_state = 10; %Minimum number of states
max_state = 10; %Maximum number of states
grid_steps = 4; %Size of increment between states

splitPercentage = 70; % Split Training Set
splitStable = 'yes'; %Options: 'yes','no'. Determines if data and test set is always identical or random
model = 'pfa'; %Options: 'hmm','pfa','dbn'
num_iter = [1 1]; %number of times EM is iterated | number of times the model will be initialized with different random values to avoid local optimum 
dataset = 'test'; %Options: 'sap','sap-small','bpi2013','test'
learn_new_model = 'yes'; %Options: 'yes','no'
prediction_mode = 'distribution'; %Options: 'simple','distribution'. 'simple' not working at the moment
draw_model = 'no'; %Options: 'yes', 'no'. Shows model of bayesian network

%% Load data set
%Input Required: only discrete attributes (except timestamp)
if strcmp(dataset,'sap')
    filename = './example/sap/SAP_P2P_COPPA_FULL.csv'; 
    delimiter = ';'; 
    timestamp_format = 'yyyy-MM-dd HH:mm:ss.SSSSSSS'; 
    CaseID = 1; Activity = 2; Timestamp = 3;    
elseif strcmp(dataset,'sap-small')
    filename = './example/sap/SAP_P2P_COPPA_SMALL.csv'; 
    delimiter = ';'; 
    timestamp_format = 'yyyy-MM-dd HH:mm:ss.SSSSSSS'; 
    CaseID = 1; Activity = 2; Timestamp = 3;    
elseif strcmp(dataset,'bpi2013')
    filename = './example/bpi2013/VINST cases closed problems.csv';
    delimiter = ';'; 
    timestamp_format = 'yyyy-MM-dd''T''HH:mm:ssXXX'; 
    CaseID = 1; Activity = 3; Timestamp = 2;
elseif strcmp(dataset,'bpi2012a')
    filename = './example/bpi2012/financial_log_application_process_ressourceContext.csv';
    delimiter = ';'; 
    timestamp_format = 'yyyy-MM-dd''T''HH:mm:ssXXX'; 
    CaseID = 1; Activity = 2; Timestamp = 3;
elseif strcmp(dataset,'test-sametrace')
    filename = './example/data_sametrace.csv';
    delimiter = ';'; 
    timestamp_format = 'yyyy-MM-dd''T''HH:mm:ssXXX'; 
    CaseID = 1; Activity = 2; Timestamp = 3;
    
else
    filename = './example/data.csv'; 
    delimiter = ';'; 
    timestamp_format = 'yyyy-MM-dd''T''HH:mm:ssXXX'; 
    CaseID = 1; Activity = 2; Timestamp = 3;
end

%Load and Prepare Data
[dataTraining dataTesting unique_values N mapping] = prepare_data(filename, delimiter, timestamp_format,CaseID,Timestamp,Activity,splitPercentage, splitStable, model); 
%% Define model and start learning
if strcmp(learn_new_model,'yes')
    % Learn new model
    [bestoverallbnet bestoverallstate] = stategrid_learning(model, N ,dataTraining,num_iter,min_state, max_state,grid_steps, unique_values);
    %save the best model on disk
    save_name = ['bestbnet_' model '_' dataset '.mat'];
    save(save_name,'bestoverallbnet');
else
    % Load existing model from disk
    disp('Loading saved model');
    load_name = ['bestbnet_' model '_' dataset '.mat'];
    load(load_name, 'bestoverallbnet');
end

%% Draw Model
if strcmp(draw_model,'yes')
    G = bestoverallbnet.dag;
    draw_graph(G);
end
%% Prediction
if strcmp(prediction_mode,'simple')
    [pred rv acc] = prediction_simple(bestoverallbnet, dataTesting);
else
    [pred rv acc] = prediction(bestoverallbnet, dataTesting);
end

prediction_ngram(dataTraining,dataTesting,unique_values);