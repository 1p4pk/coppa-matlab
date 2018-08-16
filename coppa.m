%% Import required files/functions/libraries
addpath(genpath('./data_in/'));
addpath(genpath('./libs/'));
addpath(genpath('./examples/'));
addpath(genpath('./coppa/'));

%% User Input
% State range
min_state = 5;
max_state = 5;
grid_steps = 4;

splitPercentage = 70; % Split Training Set
model = 'dbn'; %Options: 'hmm','pfa','dbn'
num_iter = [10 3]; %number of times EM is iterated | number of times the model will be initialized with different random values to avoid local optimum 
dataset = 'bpi2013'; %Options: 'sap','sap-small','bpi2013','test'
learn_new_model = 'no'; %Options: 'yes','no'
prediction_mode = 'distribution'; %Options: 'simple','distribution'

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
[dataTraining dataTesting unique_values N] = prepare_data(filename, delimiter, timestamp_format,CaseID,Timestamp,Activity,splitPercentage, model); 
%% Define model and start learning
if strcmp(learn_new_model,'yes')
    [bestoverallbnet bestoverallstate] = stategrid_learning(model, N ,dataTraining,num_iter,min_state, max_state, grid_steps,unique_values);
    disp(['Best number of states was ' num2str(bestoverallstate) '.']);
    save_name = ['bestbnet_' model '_' dataset '.mat'];
    save(save_name,'bestoverallbnet');
else
    disp('Loading saved model');
    load_name = ['bestbnet_' model '_' dataset '.mat'];
    load(load_name, 'bestoverallbnet');
end

%% Draw Model
%G = bestoverallbnet.dag;
%draw_graph(G);

%% Prediction
if strcmp(prediction_mode,'simple')
  %  [pred rv acc] = prediction_simple(bestoverallbnet, dataTesting);
else
    [pred rv acc] = prediction(bestoverallbnet, dataTesting);
end

prediction_ngram(dataTraining,dataTesting,unique_values);