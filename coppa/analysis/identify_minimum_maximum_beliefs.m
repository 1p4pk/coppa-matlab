function [] = identify_minimum_maximum_beliefs(dbnet, data, model, symptom_variables, mapping)
% IDENTIFY_MINIMUM_MAXIMUM_BELIEFS
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

disp('Start identifying minimum and maximum beliefs');

ncases = length(data);
evidence = create_evidence(dbnet, data); 

for j=1:ncases
    [ss T] = size(evidence{j});
    disp('Case:');
    disp(evidence{j});
    disp('Hypothesis variable activity')
    for i=3:length(mapping)
        n_unique = length(mapping{i});
        disp(['Context ' num2str(i)]);
        disp(['Number of unique values ' num2str(n_unique)]);
        for k=0:n_unique
            engine = bk_inf_engine(dbnet);
            evidenceToEnter = evidence{j}(:,1:T);
            if k==0
                conditional = 'only evidence';
                if strcmp(model, 'dbn')
                    evidenceToEnter(:, T) = evidenceToEnter(1,T);
                else
                    evidenceToEnter([2 symptom_variables], T) = evidenceToEnter(1,T);
                end
            else
                evidenceToEnter(:,T) = evidenceToEnter(1,T);
                evidenceToEnter(i,T) = num2cell(k);
                conditional = num2str(k);
            end
            disp('Evidence:');
            disp(evidenceToEnter);
            engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
            mA = marginal_nodes(engine, 2, T);
            disp(['Posterior distribution for ' conditional]);
            disp(mA.T);
        end
    end
end
end

