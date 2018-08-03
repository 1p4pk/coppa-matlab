function [bnet2,LLtrace] = learning(bnet,N,ncases,data_cell)
%LEARNING Summary of this function goes here
%   Detailed explanation goes here
    %% 
    %CPDs. TODO!
    
    for i=1:2*N
        bnet.CPD{i} = tabular_CPD(bnet, i);
    end

    %Example for HMM
    % Set the priors 
%    prior0 = normalise(rand(Q,1));
%    transmat0 = mk_stochastic(rand(Q,Q));
%    obsmat0 = mk_stochastic(rand(Q,O));
 %   bnet.CPD{1} = tabular_CPD(bnet, 1, prior0);
  %  bnet.CPD{2} = tabular_CPD(bnet, 2, obsmat0);
   % bnet.CPD{3} = tabular_CPD(bnet, 3, transmat0);
   
   %Example from Paper
       % Set the priors: randomly drawn from N(0,1), with diagonal covariance matrices
 %   for i = 1:(2*N)
  %      bnet.CPD{i} = gaussian_CPD(bnet, i, 'cov_type', 'diag');
  %  end
  
  
 %   for i = 1:(2*N)
	%	k = ns(i);
     %   p = 1; %If p << 1, this encourages "deterministic" CPTs (one entry near 1, the rest near 0). If p = 1, each entry is drawn from U[0,1]. If p >> 1, the entries will all be near 1/k, where k is the arity of this node, i.e., each row will be nearly uniform. 
%		ps = parents(intra, i);
%		psz = prod(ns(ps));
%		CPT = dirichlet_sample(p*ones(1,k), psz);
%		bnet.CPD{i} = tabular_CPD(bnet, i,'prior_type','dirichlet'); %, 'CPT', CPT);
 %   end
 %% 
    
    %Junction tree learning engine for parameter learning
    engine = jtree_dbn_inf_engine(bnet);

	%m = marginal_nodes(engine, nodes, t);
    
	
	max_iter=20;%iterations for EM

    %prepare structure needed. Each value of each case in a cell
    disp('Building data structure');
    cases = cell(1, ncases);
    for i=1:ncases
      T = size(data_cell{i},1);
      ev = transpose(data_cell{i}); %transpose as each column is expected to be a time slice
      cases{i} = cell(N,T);
      cases{i}(bnet.observed,:) =  num2cell(ev(bnet.observed, :)); 
    end
    
    disp('Start Learning');
	[bnet2, LLtrace] = learn_params_dbn_em(engine, cases, 'max_iter', 500);

end

