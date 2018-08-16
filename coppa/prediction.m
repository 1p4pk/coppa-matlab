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
    
    %todo include steps in usage of k in the formlas
    for k=3:T-1
        noPred = noPred + 1;
        engine = tempEngine;
        engine = enter_evidence(engine, evidence{j}(:,1:k-1));
        mS = marginal_nodes(engine, 1, k-1+steps);
        bnet = mk_bnet(dbnet.intra, dbnet.node_sizes_slice, 'discrete', dbnet.dnodes_slice, 'observed', dbnet.observed);
        for n=1:dbnet.nnodes_per_slice
            bnet.CPD{n} = dbnet.CPD{n};
        end

        bnetEngine = jtree_inf_engine(bnet);
        tempBnet = bnetEngine;
        bnetEvidence = cell(1,dbnet.nnodes_per_slice);
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