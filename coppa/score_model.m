function [acc, sens, spec] = score_model(prediction, real_values,model)
%SCORE_MODEL Summary of this function goes here
%   Detailed explanation goes here

acc = accuracy(prediction, real_values);
sens = sensitivity(prediction, real_values);
spec = specificity(prediction, real_values);

disp(['Scores for ' model ' model:']);

disp(['Accuracy of prediction: ' num2str(acc*100) '%']);
disp(['Sensitivity of prediction: ' num2str(sens*100) '%']);
disp(['Specificity of prediction: ' num2str(spec*100) '%']);

end

