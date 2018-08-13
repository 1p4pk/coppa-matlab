function [bestoverallbnet,bestoverallstate] = stategrid_learning(model, N,dataTraining,num_iter, min_state,max_state,unique_values)
%STATEGRID_LEARNING Summary of this function goes here
%   Detailed explanation goes here

bestoverallloglik = -inf; %initialize
for q=min_state:2:max_state
    disp(['Start Learning ' model ' model. State size: ' num2str(q) ', repeat ' num2str(num_iter) ' times']);
    if model=='dbn'
    bnet = create_dbn(N,unique_values,q);
    [bestbnet bestloglik] = learning(bnet,N,dataTraining,num_iter);
    elseif model=='hmm'
        bnet = create_hmm(unique_values,q);
        [bestbnet bestloglik] = learning(bnet,2,dataTraining,num_iter);
    elseif model=='pfa'
        bnet = create_pfa(unique_values,q);
        [bestbnet bestloglik] = learning(bnet,2,dataTraining,num_iter);
    end
    
    if bestloglik > bestoverallloglik
         bestoverallloglik = bestloglik;
         bestoverallbnet= bestbnet;
         bestoverallstate = q;
    end
    disp('');
end
disp('Finish learning model');
disp('');
    
end

