%COMS4771 summer B 2022
%homework 1 problem 2
%created by: Rowan Iskandar (ri2282@columbia.edu;
%rowan.iskandar@sitem-insel.ch)
%modified: 16 July 2022

xguess = 0.1;     % initial guess
emax   = 1e-9;  % Maximum error
imax   = 10000;   % maximum number of iterations
maxstep = 0.5;   % maximum x-step size
minslope = .01;  % minimum value for the slope (prevents divide by zero)
func = @(x) (x-4).^2 + 2*exp(1).^x ;
dfunc= @(x) 2*(x-4) + 2*exp(1).^x ;
lim  = @(x,xlim) max(-xlim,min(xlim,x));   % limiter function
i=1;
x=xguess;
deltax=1000;
while (i<imax) && (abs(deltax)>=emax)    % needs abs(deltax) (can be + or -)
    yguess = func(x);              % Get the function value for the x guess
    slope = dfunc(x);             % Get the function slope for x-guess
    if(abs(slope)<minslope)       % Don't allow slope to go to zero (trap divide by zero condition)
        slope = minslope*sign(slope);
    end
    yerr = yguess;                % Since the desired value is zero, the function value represents the error value 
    deltax = -yerr/slope;         % calculate the x-step
    xstep = lim(deltax,maxstep);  % limit the x-step (deltax) to +/- maxstep
    x = x + xstep;                % apply the (limited) x-step to the x-guess value and repeat the iteration
    i=i+1;                        % update the increment counter
end
