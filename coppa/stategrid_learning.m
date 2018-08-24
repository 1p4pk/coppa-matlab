function [bestoverallbnet,bestoverallstate] = stategrid_learning(model, N,dataTraining,num_iter, min_state,max_state,grid_steps,unique_values)
%STATEGRID_LEARNING
% Create model structure and start learning several times for different state values. 
% Best model in regards to log likelihood is returned
%
% Input
%%      model = type of model (e.g. "hmm" or "dbn")
%%      N = number of attributes
%%      dataTraining = data set for model learning
%%      num_iter = number of iterations for EM algorithm and number of times model is learned again to avoid local optimum (e.g. [10 5])
%%      min_state = lowest number of hidden states
%%      max_state = highest number of hidden states
%%      grid_steps = gap between tested nummer of hidden states
%%      unique_values = vector with number of unique values of each attribute
%
% Output
%%      bestoverallbnet = best learned model in regards to log likelihood
%%      bestoverallstate = number of hidden states in best model

bestoverallloglik = -inf; %initialize
for q=min_state:grid_steps:max_state % iterate through state space
    disp(['Start Learning ' model ' model. State size: ' num2str(q) ', repeat ' num2str(num_iter(2)) ' times']);
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
disp(['Best number of states was ' num2str(bestoverallstate) '.']);
end

