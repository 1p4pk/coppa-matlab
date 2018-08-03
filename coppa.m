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

%Load Data
data = import_csv(filename, delimiter); 

%Prepare Data
[dataTraining dataTesting ns datn] = prepare_data(data,timestamp_format,CaseID,Timestamp,Activity,70); 
 %% 
%Define model
N = datn -1 + 1; % number of variables in one time slice. datn + 1 (for hidden state) -1 (for case id column)
bnet = create_dbn(N,ns);
%% 

% Loop over the EM learning 100 times, keep the best model (based on
% log-likelihood), to avoid getting a model that has got stuck to a por local optimum
rng('shuffle') %init the random number generator based on time stamp
bestloglik = -inf; %initialize
for j = 1:5
    
    disp(['Starting Iteration ' num2str(j)]);
    
    %Start Learning
    
    [bnet2, LLtrace] = learning(bnet,N,dataTraining)
	
    loglik = LLtrace(length(LLtrace));
    %when we find a better model than the previous, write its results into
    %file
    if loglik > bestloglik
        bestloglik = loglik;
        bestbnet = bnet2;
    end
end

%save the bestbnet object
save('bestbnet_allHVs.mat','bestbnet')

G = bestbnet.dag;
%draw_graph(G);

prediction(bestbnet, dataTesting);
