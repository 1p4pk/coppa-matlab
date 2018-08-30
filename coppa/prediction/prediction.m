function [prediction real_value] = prediction(dbnet, data, steps)
%Calculate the prediction based on
%   dbnet = supplied dynamic bayesian network
%   ev = log case at who's end the prediction happens
%   steps = how many steps into the future
%
 if ~exist('steps','var')
      steps = 1;
 end

disp('Start Prediction');
 
ncases = length(data);
evidence = create_evidence(dbnet, data); 
prediction = cell(1,ncases);
real_value = cell(1,ncases);

for j=1:ncases
    engine = bk_inf_engine(dbnet);
    [ss T] = size(evidence{j});
    
    %In order to perform filtering, we add an empty evidence to the
    %actual evidence until t, such that we can do inference on the
    %hidden node in time slice t+steps
    evidenceToEnter = evidence{j}(:,1:T);
    %@Matthias: how to set element in cell to [] ?
    if steps > 1
        %if we want to do steps > 1 we have to adjust size of cell EvidenceToEnter, right?
        evidenceToEnter(:,T-steps:(T-1)) = evidenceToEnter(1,T-steps); 
    else
        evidenceToEnter(:,T) = evidenceToEnter(1,T);
    end
    %enter evidence and choose filtering (smoothing is default)
    engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
    %calculate marginals on the hidden node in t+steps
    mS = marginal_nodes(engine, 1, T-1+steps);

    %create static bayesian network to do inference on observed node
    %activity
    bnet = mk_bnet(dbnet.intra, dbnet.node_sizes_slice, 'discrete', dbnet.dnodes_slice, 'observed', dbnet.observed);
    for n=1:dbnet.nnodes_per_slice
        bnet.CPD{n} = dbnet.CPD{n};
    end
    bnetEngine = jtree_inf_engine(bnet);
    tempBnet = bnetEngine;
    bnetEvidence = cell(1,dbnet.nnodes_per_slice);

    %do inference on activity for each possible hidden node and weight results
    %according to probability of that hidden node
    cumulatedPred = zeros(dbnet.node_sizes_slice(2),1);
    for c=1:dbnet.node_sizes_slice(1)
        bnetEngine = tempBnet;
        bnetEvidence{1} = c;
        [bnetEngine, loglik] = enter_evidence(bnetEngine, bnetEvidence);
        mA = marginal_nodes(bnetEngine, 2);
        weightedPred = (mA.T *mS.T(c));
        cumulatedPred = cumulatedPred + weightedPred;
    end
    % cumulatedPred auf = 1 normalisieren?
    [M I] = max(cumulatedPred);
    prediction{j} = I;
    real_value{j} = evidence{j}{2,T-1+steps};
end
disp('Prediction Finished');
end