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
ss = bnet.nnodes_per_slice; %number of nodes in model
onodes = 2:ss; % all but the first two (state and activity)
evidence = create_evidence(bnet, data, onodes); %adjust to remove last event?
prediction = zeros(bnet.node_sizes_slice(1),ncases);


for j=1:ncases
    engine = bk_ff_hmm_inf_engine(bnet);
    engine = enter_evidence(engine, evidence{j});
    T = length(evidence{j});
    m = marginal_nodes(engine, 1, T);
    %get state with highest probability
    %[M I] = max(m.T);
    prediction(:,j) = m.T;
end