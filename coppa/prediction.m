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

engine = bk_ff_hmm(bnet);
engine = dbn_init_bel(engine);
ss = bnet.nnodes_per_slice; %number of nodes in model
onodes = 2:ss; % all but the first two (state and activity)
nhd = bnet.node_sizes_slice(1,1); %node size of hidden state
evidence = create_evidence(bnet, data, onodes); %adjust to remove last event?
prediction = cell(ncases);

for j=1:ncases
    T = length(evidence{j}); %length of the case
    for i=1:T
        if i>1
            engine = dbn_update_bel(engine, evidence{j}(:,i-1:i));
        else
            engine = dbn_update_bel1(engine, evidence{j}(:,i));
        end 
    end
    engine = dbn_predict_bel(engine, 1);
    m = dbn_marginal_from_bel(engine, 1);
end