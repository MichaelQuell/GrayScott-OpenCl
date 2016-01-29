%stepsizes
%This script reads the edata2,4.dat file and evaluates the convergence rate
clear all;
a2=2;
b2=12;
a4=a2;
b4=b2;
error4=importdata('data/edata43.dat');
%error2=importdata('data/edataabc.dat');
for j=[a4:b4]
p4(j)=log2(error4(j)/error4(j+1));
end;
%for j=[a2:b2]
%p2(j)=log2(error2(j)/error2(j+1));
%end;
%errorestimator
A4(:,1)=[1*2.^[-[a4-2:b4-2]]];%stepsize
A4(:,2)=error4(a4:b4);%error ||yh-yh/2||
A4(:,3)=p4(a4:b4)% aprox order
save 'data/konvergenzerrorestimator.dat' A4 '-ascii','-double';
%abc splitting
%A2(:,1)=[1*2.^[-[a2-2:b2-2]]];%stepsize
%A2(:,2)=error2(a2:b2);%error ||yh-yh/2||
%A2(:,3)=p2(a2:b2)% aprox order,2)=error2(a2:b2);%error ||yh-yh/2||
%A2(:,3)=p2(a2:b2)% aprox order
%save 'data/konvergenz.dat' A2 '-ascii','-double';