function [bestbnet bestloglik] = learning(bnet,N,data_cell,iters)
%LEARNING 
% learns model from data
%
% Input
%%      bnet = bayesian network model
%%      N = number of nodes
%%      data_cell = data for learning
%%      iters = number of iterations to avoid local optimum
%
% Output
%%  bestbnet = best learned model in regards to log likelihood
%%  bestloglik = log likelihood of best learned model

% if number of iterations is not specified use 5 as default
 if ~exist('iters','var')
      iters(2) = 5;
 end

bestloglik = -inf; %initialize
    for j = 1:iters(2)

        disp(['Starting Iteration ' num2str(j)]);

        for i=1:(N+1)
            bnet.CPD{i} = tabular_CPD(bnet, i); % add CPDs with random values
        end

        %Junction tree learning engine for parameter learning
        engine = bk_inf_engine(bnet);

        %prepare structure needed. Each value of each case in a cell
        disp('Building data structure');
        cases = create_evidence(bnet, data_cell);
        disp('Start Learning');
        [bnet2, LLtrace] = learn_params_dbn_em(engine, cases, 'max_iter', iters(1));
    	loglik = LLtrace(length(LLtrace));
    %when we find a better model than the previous, write its results into
    %file
        if loglik > bestloglik
            bestloglik = loglik;
            bestbnet = bnet2;
        end
    end
end
