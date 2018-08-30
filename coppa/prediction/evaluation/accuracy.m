function score = accuracy(prediction, real_value)
%ACCURACY 
%   Calculates prediction accuracy
%
% Input
%%      prediction = predicted values
%%      real_value = actual values
%
% Output
%%      score = accuracy of prediction

assert(isequal(size(prediction),size(real_value)),'Length of prediction vector different from vector with real values');
ncases = length(prediction);
accuracy = zeros(1,ncases);
for i=1:ncases
    if real_value{i} == prediction{i}
        accuracy(i) = 1;
    end
end
score = mean(accuracy);
end

