function [q2, q3, points] = read_solutions(filename)
%READ_SOLUTIONS Read quaternion and intersection-point solutions.
%
%   [q2, q3, points] = read_solutions(filename)
%
%   q2     is N-by-4, with columns [w x y z] for ball 2.
%   q3     is N-by-4, with columns [w x y z] for ball 3.
%   points is N-by-6-by-3. points(i,c,:) is the unit meeting point
%          for solution i and color c.
%
%   The input file may contain a comment/header line beginning with '#'.

arguments
    filename (1,1) string
end

% Read all numeric values; lines beginning with '#' are ignored.
values = readmatrix(filename, 'FileType', 'text', 'CommentStyle', '#');

if isempty(values)
    error('read_solutions:EmptyFile', 'The file contains no solution rows.');
end

if size(values, 2) ~= 26
    error('read_solutions:ColumnCount', ...
        'Expected 26 numeric columns, but found %d.', size(values, 2));
end

q2 = values(:, 1:4);
q3 = values(:, 5:8);

% Columns 9:26 are six consecutive xyz triples.
pointValues = values(:, 9:26);
points = reshape(pointValues, size(values, 1), 3, 6);
points = permute(points, [1, 3, 2]);
end
