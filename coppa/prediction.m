function [prediction real_value accuracy] = prediction(dbnet, data, steps)
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
predEventSum = 0;
%calculate number of total events in all cases
for i=1:ncases
    predEventSum = predEventSum + length(data{i}) - 3; 
    %-3 because we start predicting at 2 and dont predict from the last event on
end
evidence = create_evidence(dbnet, data); 
prediction = cell(1,predEventSum);
real_value = cell(1,predEventSum);
accuracy = zeros(1,predEventSum);

noPred = 0;
for j=1:ncases
    engine = bk_inf_engine(dbnet);
    tempEngine = engine;
    [ss T] = size(evidence{j});
    
    %todo include steps in usage of k in the formulas
    for k=2:T-steps
        noPred = noPred + 1;
        engine = tempEngine;
        %In order to perform filtering, we add an empty evidence to the
        %actual evidence until t, such that we can do inference on the
        %hidden node in time slice t+steps
        evidenceToEnter = evidence{j}(:,1:(k+steps-1));
        %@Matthias: how to set element in cell to [] ?
        if steps > 1
            evidenceToEnter(:,k:(k+steps-1)) = evidenceToEnter(1,k);
        else
            evidenceToEnter(:,k) = evidenceToEnter(1,k);
        end
        %enter evidence and choose filtering (smoothing is default)
        engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
        %calculate marginals on the hidden node in t+steps
        mS = marginal_nodes(engine, 1, k+steps-1);
        
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
        prediction{noPred} = I;
        real_value{noPred} = evidence{j}{2,k};
        if prediction{noPred} == real_value{noPred}
            accuracy(noPred) = 1;
        end
    end
end
disp('Prediction Finished');
disp(['Prediction Accuracy: ' num2str(100*mean(accuracy)) '%']);
end