addpath(genpath('./data_in/'));
addpath(genpath('./libs/'));

%Load data set
%TODO: The import only works currently for the example/data.csv file but is not generic
%Input Required: only discrete attributes
%Structure: Column 1 = CaseID, Column 2 = Activity, Column 3 = Timestamp Column 4-n = Context Attributes
data = import_csv('./example/data.csv'); 

data = sortrows(data,[1,3]) %Make sure log is sorted by CaseID and Timestamp
data(:,3) = []; %Remove Timestamp Column as it is not needed anymore

%Get info from data
[datlen datn] = size(data); %datlen = number of rows, datn = number of columns

   
% Determine dimensions for each attribute in log
unique_values = cell(1,datn)
for i=1:datn
    unique_values{i} = length(unique(data(:,i)));
end

ncases = unique_values{1}; % get number of cases from log
Q = 2; % num hidden states %input from user
unique_values{i} = Q; %replace cases count by number of states
%ns = cell2mat(unique_values);%number of states

% Split data by case, remove case id and save in cell array
 [~,~,X] = unique(data(:,1));
 data(:,1) = [];
 data_cell = accumarray(X,1:size(data,1),[],@(r){data(r,:)});
 
%Define model
N = datn -1 + 1; % number of variables in one time slice. datn + 1 (for hidden state) -1 (for case id column)
% Which nodes will be observed? 
onodes = [2:N]; %all but not the hidden state
dnodes = [1:N]; %discrete nodes = all, because we only support discrete attributes at the moment

% name the variables for easier access
State = 1; Activity = 2; 

% DAG structure: 
	
% "intra" table encodes the structure of one time slice 
intra = zeros(N,N);% table to build in the dependencies
intra(State, [Activity 3:N]) = 1; %always the same, can be created automatically. However, number of context attributes dynamic
for i=3:N
    intra(i, Activity) = 1; %all context variables influence the observation
end

% "inter" encodes the dependencies between time slices; 
inter = zeros(N,N);
inter(State,State) = 1; %all hidden variables linked to themselves across time
inter(Activity, State) = 1; % observation in t-1 infuences state in t
for i=3:N
    inter(i, State) = 1; %all context variables of t-1 influence state in t
end

% Define equivalence classes for the model variables:
%Equivalence classes are needed in order to learn the conditional
%probability tables from the data so that all data related to a variable,
%i.e. data from all years, is used to learn the distribution; the eclass
%specifies which variables "are the same".

% In the first year, all vars have their own eclasses;
% in the consecutive years, each variable belongs to the same eclass 
% with itself from the other time slices. 
% This is because due to the temporal dependencies, some of the variables have a
% different number of incoming arcs, and therefore cannot be in the same
% eclass. 

%TODO: we really have to check how this should be in our case!
eclass1 = 1:N; % first time slice
eclass2 = (N+1):(2*N);% consecutive time slices
eclass = [eclass1 eclass2];
 
% Make the model
bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2);

% Loop over the EM learning 100 times, keep the best model (based on
% log-likelihood), to avoid getting a model that has got stuck to a por local optimum
rng('shuffle') %init the random number generator based on time stamp
bestloglik = -inf; %initialize
for j = 1:5

    %CPDs. TODO!
    
    % Set the priors 
%    prior0 = normalise(rand(Q,1));
%    transmat0 = mk_stochastic(rand(Q,Q));
%    obsmat0 = mk_stochastic(rand(Q,O));
 %   bnet.CPD{1} = tabular_CPD(bnet, 1, prior0);
  %  bnet.CPD{2} = tabular_CPD(bnet, 2, obsmat0);
   % bnet.CPD{3} = tabular_CPD(bnet, 3, transmat0);
   
 %   for i = 1:(2*N)
	%	k = ns(i);
     %   p = 1; %If p << 1, this encourages "deterministic" CPTs (one entry near 1, the rest near 0). If p = 1, each entry is drawn from U[0,1]. If p >> 1, the entries will all be near 1/k, where k is the arity of this node, i.e., each row will be nearly uniform. 
%		ps = parents(intra, i);
%		psz = prod(ns(ps));
%		CPT = dirichlet_sample(p*ones(1,k), psz);
%		bnet.CPD{i} = tabular_CPD(bnet, i,'prior_type','dirichlet'); %, 'CPT', CPT);
 %   end
    
    %Junction tree learning engine for parameter learning
    engine = smoother_engine(jtree_2TBN_inf_engine(bnet));
	%engine = enter_evidence(engine, data);
	%m = marginal_nodes(engine, nodes, t);
    
	
	max_iter=20;%iterations for EM

    %prepare structure needed. Each value of each case in a cell
    cases = cell(1, ncases);
    for i=1:ncases
      T = length(data_cell{i});
      cases{i} = cell(N,T);
      cases{i}(onodes,:) =  cellstr(transpose(data_cell{i}));
    end

	[bnet2, LLtrace] = learn_params_dbn_em(engine, cases, 'max_iter', 500);
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
draw_graph(G);