function [translated_value] = map_to_name(number_attr, numeric_value,mapping_table)
%MAP_TO_NAME Summary of this function goes here
%   Detailed explanation goes here
    translated_value = mapping_table{number_attr}{numeric_value};
end

