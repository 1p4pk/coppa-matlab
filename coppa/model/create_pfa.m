function [bnet] = create_pfa(unique_values,Q)
%CREATE_DBN Summary of this function goes here
%   Detailed explanation goes here
N = 2;
%Q = 4; % num hidden states %input from user
unique_values{1} = Q; %replace cases count by number of states
ns = cell2num(unique_values(1:2));%number of states

% Which nodes will be observed? 
onodes = [2]; %all but not the hidden state
dnodes = [1:2]; %discrete nodes = all, because we only support discrete attributes at the moment

% name the variables for easier access
State = 1; Activity = 2; 

% DAG structure: 
	
% "intra" table encodes the structure of one time slice 
intra = zeros(N,N);% table to build in the dependencies
intra(State, Activity) = 1; %always the same, can be created automatically. However, number of context attributes dynamic

% "inter" encodes the dependencies between time slices; 
inter = zeros(N,N);
inter(State,State) = 1; %all hidden variables linked to themselves across time
inter(Activity,State) = 1;

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


eclass1 = [1 2]; % first time slice, all nodes have their own eclass
%TODO: we really have to check how this should be in our case!
%see http://bayesnet.github.io/bnt/docs/usage.html#tying and http://bayesnet.github.io/bnt/docs/usage_dbn.html#hmm
eclass2 = [3 2];% consecutive time slices,
eclass = [eclass1 eclass2];
 

%Add node names to beautify
names = {};
names = [names;'Hidden State'];
names = [names;'Observation'];

% Make the model
bnet = mk_dbn(intra, inter, ns, 'discrete', dnodes,'observed',onodes, 'eclass1', eclass1, 'eclass2', eclass2,'names',names);
end

