function [prediction real_value accuracy] = prediction(bnet, data, steps)
%Calculate the prediction based on
%   bnet = supplied dynamic bayesian network
%   ev = log case at who's end the prediction happens
%   steps = how many steps into the future
%
 if ~exist('steps','var')
      steps = 1;
 end

disp('Start Prediction');
 
ncases = length(data);
evidence = create_evidence(bnet, data); 
prediction = cell(1,ncases);
real_value = cell(1,ncases);
accuracy = zeros(1,ncases);


for j=1:ncases
    engine = bk_inf_engine(bnet);
    [ss T] = size(evidence{j});
    temp = engine;
    engine = enter_evidence(engine, evidence{j}(:,1:T-1));
    mS = marginal_nodes(engine, 1, T-1+steps);
    cumulatedPred = zeros(bnet.node_sizes_slice(2),1);
    for c=1:bnet.node_sizes_slice(1)
        engine = temp;
        new_evidence = evidence{j};
        for z=1:ss
            new_evidence{z,T} = [];
        end
        new_evidence{1,T} = c; 
        engine = enter_evidence(engine, new_evidence);
        mA = marginal_nodes(engine, 2, T-1+steps);
        weightedPred = (mA.T *mS.T(c));
        cumulatedPred = cumulatedPred + weightedPred;
    end
    % cumulatedPred auf = 1 normalisieren?
    [M I] = max(cumulatedPred);
    prediction{j} = I;
    real_value{j} = evidence{j}{2,T};
    if prediction{j} == real_value{j}
        accuracy(j) = 1;
    end
    
end
disp('Prediction Finished');
disp(['Prediction Accuracy: ' num2str(100*mean(accuracy)) '%']);
end