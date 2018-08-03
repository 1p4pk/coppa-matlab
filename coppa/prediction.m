function prediction = prediction(bnet, ev, steps)
%Calculate the prediction based on
%   bnet = supplied dynamic bayesian network
%   ev = log case at who's end the prediction happens
%   steps = how many steps into the future
%

engine = jtree_dbn_inf_engine(bnet);
T = length(ev); %length of the case
ss = bnet.nnodes_per_slice; %number of nodes in model
onodes = 2:ss; % all but the first two (state and activity)
nhd = bnet.node_sizes_slice(1,1); %node size of hidden state

for i=1:nhd
    evidence = cell(ss,T);
    evidence(onodes,:) = num2cell(ev(onodes, :)); % all cells besides onodes are empty
    [engine, ll] = enter_evidence(engine, evidence);
    marg = marginal_nodes(engine, i, T+steps); %calculate marginal nodes for hidden state i for steps into the future
end

prediction = marg;