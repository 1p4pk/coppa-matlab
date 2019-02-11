function [bnet] = create_dbn_new(N,unique_values,Q,background_variables,symptom_variables)
%CREATE_DBN Summary of this function goes here
%   Detailed explanation goes here
%Q = 4; % num hidden states %input from user
unique_values{1} = Q; %replace cases count by number of states
ns = cell2mat(unique_values);%number of states

% Which nodes will be observed? 
onodes = [2:N]; %all but not the hidden state
dnodes = [1:N]; %discrete nodes = all, because we only support discrete attributes at the moment

% name the variables for easier access
State = 1; Activity = 2; 

% DAG structure: 
	
% "intra" table encodes the structure of one time slice 
intra = zeros(N,N);% table to build in the dependencies
intra(State, [Activity background_variables]) = 1; %always the same, can be created automatically. However, number of context attributes dynamic
intra(background_variables, Activity) = 1;
intra(Activity, symptom_variables) = 1; %all observations influence symptom variables

% "inter" encodes the dependencies between time slices; 
inter = zeros(N,N);
inter(State,State) = 1; %all hidden variables linked to themselves across time
inter(Activity, State) = 1; % observation in t-1 infuences state in t
inter(symptom_variables, State) = 1;
inter(background_variables, State) = 1; %all context variables of t-1 influence state in t

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


eclass1 = 1:N; % first time slice, all nodes have their own eclass
%TODO: we really have to check how this should be in our case!
%see http://bayesnet.github.io/bnt/docs/usage.html#tying and http://bayesnet.github.io/bnt/docs/usage_dbn.html#hmm
eclass2 = [(N+1),2:N];% consecutive time slices,
eclass = [eclass1 eclass2];
 

%Add node names to beautify
pre = 'Context';
names = {};
names = [names;'Hidden State'];
names = [names;'Observation'];
for k = 1:N-2
    names = [names;strcat([pre,num2str(k)])];
end

% Make the model
bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes,'observed',onodes, 'eclass1', eclass1, 'eclass2', eclass2,'names',names);
end

