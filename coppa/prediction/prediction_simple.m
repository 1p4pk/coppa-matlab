function [prediction real_value accuracy] = prediction_simple(bnet, data, steps)
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
    mS = marginal_nodes(engine, 1, T-1+steps); %T-1 as we are predict the event in T to compare it to the event in the log in T
    [M I] = max(mS.T);
    engine = temp;
    new_evidence = evidence{j};
    for z=1:ss
        new_evidence{z,T} = [];
    end
    new_evidence{1,T} = I; 
    engine = enter_evidence(engine, new_evidence);
    mA = marginal_nodes(engine, 2, T-1+steps);
    [M2 I2] = max(mA.T);
    prediction{j} = I2;
    real_value{j} = evidence{j}{2,T};
    if prediction{j} == real_value{j}
        accuracy(j) = 1;
    end
    
end
disp('Prediction Finished');
disp(['Prediction Accuracy: ' num2str(100*mean(accuracy)) '%']);
end