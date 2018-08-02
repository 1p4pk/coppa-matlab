N = 4; % number of variables in one time slice (2 + number of contextual attributes. Can be derived dynamically from the data set)

% name the variables for easier access
State = 1; Activity = 2; Context1 = 3; %use column names from data?
Context2 = 4; 

% DAG structure: 
	
% "intra" table encodes the structure of one time slice 
intra = zeros(N,N);% table to build in the dependencies
intra(State, [Activity Context1 Context2]) = 1; %always the same, can be created automatically. However, number of context attributes dynamic
intra(Context1, Activity) = 1; %always the same, can be created automatically
intra(Context2, Activity) = 1; %always the same, can be created automatically

% "inter" encodes the dependencies between tme slices; 
inter = zeros(N,N);
inter(State,State) = 1; %all hidden variables linked to themselves across time
inter(Activity, State) = 1; 
inter(Context1, State) = 1; %always the same, can be created automatically
inter(Context2, State) = 1; %always the same, can be created automatically

% Read in the data. 
data = import_csv('data.csv');
data = num2cell(data);
[datlen datn] = size(data); 

% Which nodes will be observed? 
onodes = [2:N]; 
dnodes = [1:N]; 

Q = 5; % num hidden states %input from user
O = 4; % num observations % get from log
C1 = 3; % num resources % get from log
C2 = 2; % num critical % get from log
ns = [Q O C1 C2];%number of states

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

eclass1 = 1:N; % first time slice
eclass2 = (N+1):(2*N);% consecutive time slices
eclass = [eclass1 eclass2];
 
% Make the model
bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2);

% Loop over the EM learning 100 times, keep the best model (based on
% log-likelihood), to avoid getting a model that has got stuck to a por local optimum
%rng('shuffle') %init the random number generator based on time stamp
bestloglik = -inf; %initialize
for j = 1:5

    % Set the priors 
    prior0 = normalise(rand(Q,1));
    transmat0 = mk_stochastic(rand(Q,Q));
    obsmat0 = mk_stochastic(rand(Q,O));
    bnet.CPD{1} = tabular_CPD(bnet, 1, prior0);
    bnet.CPD{2} = tabular_CPD(bnet, 2, obsmat0);
    bnet.CPD{3} = tabular_CPD(bnet, 3, transmat0);
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
    
	%data = [xpos(:)'; ypos(:)']; 
	ncases = 3 % number of cases in log % get from log
	T=4; % number of time slices, differs for each case!
	max_iter=20;%iterations for EM
	cases = cell(1, ncases);
	onodes = bnet.observed;
	for i=1:ncases
	  cases{i} = cell(N,T);
	  cases{i}(onodes,:) = data(onodes, :);
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