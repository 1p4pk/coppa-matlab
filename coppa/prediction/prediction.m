function [prediction real_value prediction_probabilities] = prediction(dbnet, data, symptom_variables, steps)
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
prediction_probabilities = cell(dbnet.node_sizes_slice(2),ncases);
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
        evidenceToEnter([2 symptom_variables], T) = evidenceToEnter(1,T);
    end
    %enter evidence and choose filtering (smoothing is default)
    engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
    %calculate marginals of the activity node in t+steps
    mA = marginal_nodes(engine, 2, T-1+steps);

    [M I] = max(mA.T);
    prediction{j} = I;
    %@Matthias, how to properly construct this cell of different activity
    %predictions?
    prediction_probabilities{j}(:,j) = mA.T; 
    real_value{j} = evidence{j}{2,T-1+steps};
end
disp('Prediction Finished');
end