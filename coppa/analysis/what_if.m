function [] = what_if(dbnet, data, model, symptom_variables, mapping, N)
% WHAT_IF_ANALYSIS
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

disp('Start what-if analysis');

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
    for i=1:N
        n_unique = length(mapping{i+2});
        disp(['Context: ' num2str(i)])
        for k=1:n_unique
            engine = bk_inf_engine(dbnet);    
            evidenceToEnter = evidence{j}(:,1:T);
            evidenceToEnter([2 symptom_variables], T) = evidenceToEnter(1,T);
            evidenceToEnter(i+2, T-1) = num2cell(k);
            disp(['State: ' num2str(k)]);
            disp('Evidence:');
            disp(evidenceToEnter);
            engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
            mA = marginal_nodes(engine, 2, T);
            disp(['P(' mapping{2}{real_v} '| y=' num2str(k) ') = ' num2str(mA.T(real_v))])
        end
    end
end
end

