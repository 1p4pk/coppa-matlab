function [min, normal, max] = identify_minimum_maximum_beliefs(dbnet, data, model, symptom_variables, mapping)
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
min = [];
max = [];
normal = [];

for j=1:ncases
    [ss T] = size(evidence{j});

    n_unique = length(mapping{3});
    values_min = ones(1,length(mapping{2}));
    values_max = zeros(1,length(mapping{2}));
    for k=0:n_unique
        
        engine = bk_inf_engine(dbnet);
        evidenceToEnter = evidence{j}(:,1:T);
        evidenceToEnter(:, T) = evidenceToEnter(1,T);
        if k>0
           evidenceToEnter(3, T) = num2cell(k); 
        end
        
        engine = enter_evidence(engine, evidenceToEnter, 'filter', 1);
        mA = marginal_nodes(engine, 2, T);
        
        if k==0
           normal = [normal; reshape(mA.T, [1, length(mA.T)])];
        else
           for i=1:length(mapping{2})
               if values_min(1,i) > mA.T(i)
                   values_min(1,i) = mA.T(i);
               end
               if values_max(1,i) < mA.T(i)
                   values_max(1,i) = mA.T(i);
               end
           end    
        end
        
    end
    min = [min; values_min];
    max = [max; values_max];
end
end

