function [result] = impact_of_evidence(dbnet, data, model, symptom_variables, mapping)
% IMPACT_OF_EVIDENCE_SUBSETS
% Identify minimum and maximum beliefs for activity given all possible
% observations on the context variables and evidence.
%
% Input
%%      dbnet = supplied dynamic bayesian network
%%      data = data to do analysis with
%%      model = type of model (e.g. "dbn_new" or "dbn")
%%      symptom_variables = context variables
%%      mapping = actual values for numbers in evidence
%
% Output
%%      no output

disp('Start identifying impact of evidence subsets');

ncases = length(data);
evidence = create_evidence(dbnet, data);
result = [];

for j=1:ncases
    [ss T] = size(evidence{j});
    
    % Assume that we we observe the activity to be the actual real value
    real_v = cell2num(evidence{j}(2,T));
    
    engine = bk_inf_engine(dbnet);
    evidenceToEnter = evidence{j}(:,1:T);
    evidenceToEnter(:,T-1:T) = evidenceToEnter(1,T);
    engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
    mA = marginal_nodes(engine, 2, T);
    apriori = mA.T(real_v);
    values = [];
%     disp('Evidence: A CT-1');
    
%     disp('Evidence: 1  0');
    engine = bk_inf_engine(dbnet);
    evidenceToEnter = evidence{j}(:,1:T);
    evidenceToEnter(3,T-1) = evidenceToEnter(1,T);
    evidenceToEnter(:,T) = evidenceToEnter(1,T);
    engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
    mA = marginal_nodes(engine, 2, T);
    if isnan(mA.T(real_v)/apriori)
        values = [values, 0];
    else
        values = [values, mA.T(real_v)/apriori];
    end
  
    %     disp('Evidence: 0  1');
    engine = bk_inf_engine(dbnet);
    evidenceToEnter = evidence{j}(:,1:T);
    evidenceToEnter(2,T-1) = evidenceToEnter(1,T);
    evidenceToEnter(:,T) = evidenceToEnter(1,T);
    engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
    mA = marginal_nodes(engine, 2, T);
    if isnan(mA.T(real_v)/apriori)
        values = [values, 0];
    else
        values = [values, mA.T(real_v)/apriori];
    end
    
%     disp('Evidence: 1  1');
    engine = bk_inf_engine(dbnet);
    evidenceToEnter = evidence{j}(:,1:T);
    evidenceToEnter(:,T) = evidenceToEnter(1,T);
    engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
    mA = marginal_nodes(engine, 2, T);
    if isnan(mA.T(real_v)/apriori)
        values = [values, 0];
    else
        values = [values, mA.T(real_v)/apriori];
    end
    result = [result;values];
end
disp('Finished identifying impact of evidence subsets');
end

