function write_config(nn, fname)
%WRITE_CONFIG Write an 18-by-3 plane-normal configuration file.
%
% The rows are written in their input order: six normals for ball 1,
% followed by six normals for ball 2 and six normals for ball 3.

if ~isnumeric(nn) || ~isreal(nn) || ~isequal(size(nn), [18, 3])
    error('write_config:InvalidNormals', ...
        'nn must be a real numeric matrix with size 18-by-3.');
end

if ~(ischar(fname) || (isstring(fname) && isscalar(fname)))
    error('write_config:InvalidFilename', ...
        'fname must be a character vector or scalar string.');
end

file_id = fopen(fname, 'w');
if file_id == -1
    error('write_config:OpenFailed', ...
        'Could not open file for writing: %s', char(fname));
end

cleanup = onCleanup(@() fclose(file_id));

fprintf(file_id, ...
    '# ball1 colors 1..6, ball2 colors 1..6, ball3 colors 1..6 (plane normals)\n');

for row = 1:18
    fprintf(file_id, '%+.10f %+.10f %+.10f\n', nn(row, 1), nn(row, 2), nn(row, 3));
end

clear cleanup;
end