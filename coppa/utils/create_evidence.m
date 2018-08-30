function [cases] = create_evidence(bnet,data_cell,onodes)
%CREATE_EVIDENCE Summary of this function goes here
%   Detailed explanation goes here

 if ~exist('onodes','var')
      onodes = bnet.observed;
 end

ncases = length(data_cell);
    
cases = cell(1, ncases);
N = bnet.nnodes_per_slice;

    for i=1:ncases
      T = size(data_cell{i},1);
      ev = transpose(data_cell{i}); %transpose as each column is expected to be a time slice
      cases{i} = cell(N,T);
      cases{i}(onodes,:) =  num2cell(ev(onodes, :)); 
    end
    
end

