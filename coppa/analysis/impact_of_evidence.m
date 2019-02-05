function [] = impact_of_evidence(dbnet, data, model, symptom_variables, mapping)
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

for j=1:ncases
    [ss T] = size(evidence{j});
    disp('Case:');
    disp(evidence{j});
    disp('Hypothesis variable activity:')
    % Assume that we we observe the activity to be the actual real value
    real_v = cell2num(evidence{j}(2,T));
    disp(['Hypothesis: Activity = ' mapping{2}{real_v}]);
    engine = bk_inf_engine(dbnet);
    evidenceToEnter = evidence{j}(:,1:T);
    evidenceToEnter(:,T-1:T) = evidenceToEnter(1,T);
    engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
    mA = marginal_nodes(engine, 2, T);
    disp(['P(' mapping{2}{real_v} ') = ' num2str(mA.T(real_v))])
    disp('Evidence: A CT-1 CT');
    
    disp('Evidence: 0  0   1');
    engine = bk_inf_engine(dbnet);
    evidenceToEnter = evidence{j}(:,1:T);
    evidenceToEnter(:,T-1) = evidenceToEnter(1,T);
    evidenceToEnter(2,T) = evidenceToEnter(1,T);
    engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
    mA = marginal_nodes(engine, 2, T);
    disp(['P(' mapping{2}{real_v} '|CT) = ' num2str(mA.T(real_v))])
    
    disp('Evidence: 0  1   0');
    engine = bk_inf_engine(dbnet);
    evidenceToEnter = evidence{j}(:,1:T);
    evidenceToEnter(2,T-1) = evidenceToEnter(1,T);
    evidenceToEnter(:,T) = evidenceToEnter(1,T);
    engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
    mA = marginal_nodes(engine, 2, T);
    disp(['P(' mapping{2}{real_v} '|CT-1) = ' num2str(mA.T(real_v))])
    
    disp('Evidence: 0  1   1');
    engine = bk_inf_engine(dbnet);
    evidenceToEnter = evidence{j}(:,1:T);
    evidenceToEnter(2,T-1:T) = evidenceToEnter(1,T);
    engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
    mA = marginal_nodes(engine, 2, T);
    disp(['P(' mapping{2}{real_v} '|CT-1,CT) = ' num2str(mA.T(real_v))])
    
    disp('Evidence: 1  0   0');
    engine = bk_inf_engine(dbnet);
    evidenceToEnter = evidence{j}(:,1:T);
    evidenceToEnter(3,T-1) = evidenceToEnter(1,T);
    evidenceToEnter(:,T) = evidenceToEnter(1,T);
    engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
    mA = marginal_nodes(engine, 2, T);
    disp(['P(' mapping{2}{real_v} '|A) = ' num2str(mA.T(real_v))])
    
    disp('Evidence: 1  0   1');
    engine = bk_inf_engine(dbnet);
    evidenceToEnter = evidence{j}(:,1:T);
    evidenceToEnter(3,T-1) = evidenceToEnter(1,T);
    evidenceToEnter(2,T) = evidenceToEnter(1,T);
    engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
    mA = marginal_nodes(engine, 2, T);
    disp(['P(' mapping{2}{real_v} '|A,CT) = ' num2str(mA.T(real_v))])
    
    disp('Evidence: 1  1   0');
    engine = bk_inf_engine(dbnet);
    evidenceToEnter = evidence{j}(:,1:T);
    evidenceToEnter(:,T) = evidenceToEnter(1,T);
    engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
    mA = marginal_nodes(engine, 2, T);
    disp(['P(' mapping{2}{real_v} '|A,CT-1) = ' num2str(mA.T(real_v))])
    
    disp('Evidence: 1  1   1');
    engine = bk_inf_engine(dbnet);
    evidenceToEnter = evidence{j}(:,1:T);
    evidenceToEnter(2,T) = evidenceToEnter(1,T);
    engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
    mA = marginal_nodes(engine, 2, T);
    disp(['P(' mapping{2}{real_v} '|A,CT-1,CT) = ' num2str(mA.T(real_v))])
end
end

