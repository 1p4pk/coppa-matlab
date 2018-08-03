function prediction = prediction(bnet, data, steps)
%Calculate the prediction based on
%   bnet = supplied dynamic bayesian network
%   ev = log case at who's end the prediction happens
%   steps = how many steps into the future
%
 if ~exist('steps','var')
      steps = 1;
 end

ncases = length(data);

engine = jtree_dbn_inf_engine(bnet);
ss = bnet.nnodes_per_slice; %number of nodes in model
onodes = 2:ss; % all but the first two (state and activity)
nhd = bnet.node_sizes_slice(1,1); %node size of hidden state
evidence = create_evidence(bnet, data, onodes); %adjust to remove last event?
prediction = cell(ncases);

for j=1:ncases
    T = length(evidence{j}); %length of the case
    for i=1:nhd
          [engine, ll] = enter_evidence(engine, evidence{j});
          marg = marginal_nodes(engine, i, T+steps); %calculate marginal nodes for hidden state i for steps into the future 
    end
end