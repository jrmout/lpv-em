function clearall

% Clear all variables but not breakpoints. Useful
% as a substitute for "clear all" when debugging.
%
%

breakpoints = dbstatus('-completenames');
evalin('base', 'clear classes');
dbstop(breakpoints);
