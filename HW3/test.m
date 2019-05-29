clear;

Q = [0.8,-0.6;0.6,0.8];
Q_1 = inv(Q);
lambda = [0.4,0;0,0.9];

W_norm = [];
% W = [];

for x = 1:100
	% W(x) =;
	W_norm(x) = norm(Q*power(lambda,x)*Q_1,2);
end


plot(1:100,W_norm);
