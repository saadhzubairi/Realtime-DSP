b0 = 1.0;
G  = 0.8;
N  = 5;

b = [b0 zeros(1,N-1) G];  % FIR numerator
a = 1;                    % Denominator

figure;
zplane(b,a);
title('Pole-Zero Plot of Echo FIR Filter');
