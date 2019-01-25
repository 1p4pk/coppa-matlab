function [acc, sens, spec] = score_model(prediction, real_values)
%SCORE_MODEL Summary of this function goes here
%   Detailed explanation goes here

acc = accuracy(prediction, real_values);
sens = sensitivity(prediction, real_values);
spec = specificity(prediction, real_values);

end

