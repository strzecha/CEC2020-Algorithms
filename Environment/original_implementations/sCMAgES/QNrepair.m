function [x,it]  = QNrepair(problem, x0,g,h)
xmin             = problem.xmin(:);
xmax             = problem.xmax(:);
fhd              = @objf;
if problem.gn == 0
    x0        = [x0(:); sqrt(x0(:)-xmin); sqrt(xmax-x0(:))];
else
    x0        = [x0(:); sqrt(x0(:)-xmin); sqrt(xmax-x0(:)); sqrt(max(0,-g(:)))];
end
options.maxiter    = problem.maxiter;
options.display    = 0;

[x,y,it] = OGN(fhd,x0,options,problem);

if sum(real(x) ~= x)
    x = x0(1:problem.n);
else
    x = x(1:problem.n);
end
end


function [x,y,it] = OGN(fhd,x0,opt,problem)
lambda = 0.01;
I = eye(length(x0));
Z = zeros(length(x0),1);
x = x0;
[F,J] = feval(fhd,x,problem);
it = 1;
%% stopping criteria parameters
tolF    = 1e-25;
tolX    = 1e-12;
sqrtEps = sqrt(eps);
tolOpt  = 1e-4*tolF;
relFactor = max(norm(J.'*F,inf),sqrtEps);
%% iteration parameters
stop = 0;
flag  = 0;
%% iteration starts
while (stop == 0)
d  = [J;sqrt(lambda)*I]\[F;Z];
trialx = x-d;
[trialF,trialJ] = feval(fhd,trialx,problem);
it = it+1;
if norm(trialF) < norm(F)
    if flag == 0
        lambda = lambda/10;
    end
    x = trialx;
    oldF = F;
    F = trialF;
    J = trialJ;
    %% stopping criteria
    if (abs(F'*F-oldF'*oldF) < tolF*(oldF'*oldF))||( it > opt.maxiter)|| (norm(d) < tolX*(sqrtEps+norm(x))) || (norm(J'*F,inf) < tolOpt*relFactor) || (flag > 10)
        stop = 1;
    end
    flag = 0;
if opt.display == 1
    disp([num2str(it),'    m=', num2str(lambda), '------> ', num2str(max(norm(F)^2)),' TolF = ',num2str(norm(J'*F,inf)), ' TolX = ', num2str(norm(d))]);
end
else
    flag = flag + 1;
    lambda = 10*lambda;

end
    if (flag > 10)||( it > opt.maxiter)
        stop = 1;
    end
end

y = norm(F);
end


function [f,J] = objf(xx,problem)
J             = Jocob_Ana(xx,problem);
xmin          = problem.xmin(:);
xmax          = problem.xmax(:);
x             = xx(1:problem.n);
lb            = xx(problem.n+1:2*problem.n);lb = lb(:);
ub            = xx(2*problem.n+1:3*problem.n); ub = ub(:);
gb            = xx(3*problem.n+1:end); gb = gb(:);
yb            = [x-lb.^2-xmin;x+ub.^2-xmax];
[~,g,h] = feval(problem.constr_fun_name,x',problem.I_fno);
g = g(:)+gb.^2;
if problem.gn == 0
    f = [h(:);yb];
elseif problem.hn == 0
    f = [g(:);yb];
else
f = [g(:);h(:);yb];
end
end
