function stop = maxtime(x,optimValues,state,iterstarttime,max_cpu_time)

% MAXTIME is used to stop NLP solver fmincon whenever the maximum
% CPU time s exceeded.
%
% See also FMINSDP

stop = false;
if toc(iterstarttime)>max_cpu_time
    fprintf('\n\nfmincon stopped because the maximum CPU time (%d [s]) was exceeded. \n\n',max_cpu_time);
    stop = true;
end
